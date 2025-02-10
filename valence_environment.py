import os
import neat
import numpy as np
from datetime import datetime

class ValenceEnvironment:
    def __init__(self, size=32):
        self.size = size
        self.grid = np.zeros((size, size, 3))
        
        # Keep zones for organization
        self.zones = {
            'gradient_zone': (0, 0, size//2, size),
            'conflict_zone': (size//2, 0, size, size)
        }
        
        # Separate dynamic rewards from static punishments
        self.active_reward = None
        self.points = {
            'static_punishments': []
        }
        
        # Sensory parameters
        self.max_sensory_distance = 15  # How far the agent can "see"
        self.reward_collection_radius = 2  # How close to get to collect reward
        
        self.setup_environment()
    
    def setup_environment(self):
        # exactly 2 static punishments
        self.points['static_punishments'] = [
            {
                'pos': [np.random.randint(0, self.size), np.random.randint(0, self.size)],
                'value': -0.5,
                'radius': 4
            },
            {
                'pos': [np.random.randint(0, self.size), np.random.randint(0, self.size)],
                'value': -0.5,
                'radius': 4
            }
        ]
        
        # Setup initial reward
        self.spawn_new_reward()
    
    def spawn_new_reward(self):
        """Create a new reward at a random location"""
        while True:
            # Try to place reward away from punishments
            pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
            valid = True
            
            # Check distance from punishments
            for punct in self.points['static_punishments']:
                dist = np.linalg.norm(np.array(pos) - np.array(punct['pos']))
                if dist < punct['radius'] * 2:
                    valid = False
                    break
            
            if valid:
                self.active_reward = {
                    'pos': pos,
                    'value': 1.0,
                    'radius': 8
                }
                break
    
    def get_sensory_input(self, pos):
        """Get sensory information with continuous gradients from -1 (punishment) to 1 (reward)"""
        inputs = []
        directions = [0, np.pi/2, np.pi, 3*np.pi/2]  # 4 directions
        
        for angle in directions:
            direction = np.array([np.cos(angle), np.sin(angle)])
            net_valence = 0  # Combined reward/punishment signal
            
            # Cast rays
            for dist in np.linspace(1, self.max_sensory_distance, 15):
                point = pos + direction * dist
                if 0 <= point[0] < self.size and 0 <= point[1] < self.size:
                    # Get reward gradient (positive values)
                    if self.active_reward:
                        r_dist = np.linalg.norm(point - np.array(self.active_reward['pos']))
                        if r_dist < self.max_sensory_distance:
                            # Exponential falloff mapped to 0 to 1
                            intensity = np.exp(-r_dist / self.max_sensory_distance)
                            net_valence += intensity
                    
                    # Get punishment gradient (negative values)
                    for punct in self.points['static_punishments']:
                        p_dist = np.linalg.norm(point - np.array(punct['pos']))
                        if p_dist < punct['radius'] * 2:
                            intensity = np.exp(-p_dist / (punct['radius'] * 2))
                            net_valence -= intensity
            
            # Clip final value between -1 and 1
            inputs.append(np.clip(net_valence, -1, 1))
        
        return inputs  # Now returns 4 inputs (one per direction) instead of 8
    
    def update(self, agent_pos):
        """Update environment state and return reward"""
        reward = 0
        
        # Check if agent reached reward
        if self.active_reward:
            dist_to_reward = np.linalg.norm(np.array(agent_pos) - np.array(self.active_reward['pos']))
            if dist_to_reward < self.reward_collection_radius:
                reward = self.active_reward['value']
                self.spawn_new_reward()
        
        # Calculate punishment
        punishment = 0
        for punct in self.points['static_punishments']:
            dist = np.linalg.norm(np.array(agent_pos) - np.array(punct['pos']))
            if dist < punct['radius']:
                punishment += abs(punct['value']) * (1 - dist/punct['radius'])
        
        return reward - punishment
    
    def evaluate_navigation(self, positions, movements):
        """Simplified evaluation metrics"""
        if len(positions) == 0:
            return {
                'gradient_following': 0,
                'conflict_resolution': 0,
                'efficiency': 0
            }
        
        gradient_score = self.evaluate_gradient_following(positions)
        conflict_score = self.evaluate_conflict_resolution(positions, movements)
        efficiency_score = self.evaluate_movement_efficiency(positions, movements)
        
        return {
            'gradient_following': gradient_score,
            'conflict_resolution': conflict_score,
            'efficiency': efficiency_score
        }
    
    def evaluate_gradient_following(self, positions):
        """Simplified gradient following evaluation"""
        score = 0
        positions = np.array(positions)
        
        for i in range(1, len(positions)):
            prev_reward, prev_punishment = self.get_valence_at_point(positions[i-1])
            curr_reward, curr_punishment = self.get_valence_at_point(positions[i])
            
            score += (curr_reward - prev_reward) + (prev_punishment - curr_punishment)
            
        return score / len(positions)
    
    def evaluate_conflict_resolution(self, positions, movements):
        """Simplified conflict resolution evaluation"""
        if not len(movements):
            return 0
            
        score = 0
        positions = np.array(positions)
        
        for i, pos in enumerate(positions):
            if i >= len(movements):
                break
                
            rewards, punishments = self.get_valence_at_point(pos)
            movement_mag = np.linalg.norm(movements[i])
            
            if rewards > 0 and punishments > 0:
                if movement_mag > 0.5:  # Decisive movement in conflict
                    score += 1
                    
        return score / len(positions)
    
    def evaluate_movement_efficiency(self, positions, movements):
        """Simplified movement efficiency evaluation"""
        if not len(movements):
            return 0
            
        smoothness_score = 0
        for i in range(1, len(movements)):
            try:
                dot_product = np.dot(movements[i], movements[i-1])
                mags_product = np.linalg.norm(movements[i]) * np.linalg.norm(movements[i-1])
                if mags_product > 0:
                    angle = np.arccos(np.clip(dot_product / mags_product, -1, 1))
                    smoothness_score += 1 - angle/np.pi
            except:
                continue
                
        return smoothness_score / (len(movements) - 1) if len(movements) > 1 else 0