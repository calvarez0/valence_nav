# network_analysis.py

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
import neat
from neat.graphs import feed_forward_layers

class NetworkAnalyzer:
    def __init__(self):
        """Initialize the analyzer with empty logs"""
        self.reset_logs()
    
    def reset_logs(self):
        """Reset all logs to empty state"""
        self.behavior_log = []
        self.network_states = []
        self.movement_log = []
    
    def analyze_network(self, genome, config):
        """Analyze network structure and create graph visualization"""
        network = nx.DiGraph()
        
        # Add nodes
        input_nodes = config.genome_config.input_keys
        output_nodes = config.genome_config.output_keys
        hidden_nodes = [n for n in genome.nodes.keys() if n not in input_nodes and n not in output_nodes]
        
        # Add edges with weights
        for cg in genome.connections.values():
            if cg.enabled:
                network.add_edge(cg.key[0], cg.key[1], weight=cg.weight)
        
        return network
    
    def analyze_valence_processing(self, genome, config):
        """Analyze how the network processes valence signals"""
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Initialize arrays to store responses
        # Assuming 2 output nodes (dx, dy) based on your environment
        n_distances = 10
        n_angles = 8
        reward_responses = np.zeros((n_distances * n_angles, 4))  # [distance, angle, out_x, out_y]
        punishment_responses = np.zeros((n_distances * n_angles, 4))  # [distance, angle, out_x, out_y]
        
        idx = 0
        # Test reward response (varying distance/direction)
        for distance in np.linspace(0, 1, n_distances):
            for angle in np.linspace(0, 2*np.pi, n_angles):
                # Create 4 inputs (4 directions with net valence)
                inputs = [0] * 4
                direction_idx = int((angle / (2*np.pi) * 4)) % 4
                inputs[direction_idx] = distance  # Set positive valence for this direction
                
                output = net.activate(inputs)
                reward_responses[idx] = [distance, angle, output[0], output[1]]
                idx += 1
        
        idx = 0
        # Test punishment response
        for distance in np.linspace(0, 1, n_distances):
            for angle in np.linspace(0, 2*np.pi, n_angles):
                inputs = [0] * 4
                direction_idx = int((angle / (2*np.pi) * 4)) % 4
                inputs[direction_idx] = -distance  # Set negative valence for this direction
                
                output = net.activate(inputs)
                punishment_responses[idx] = [distance, angle, output[0], output[1]]
                idx += 1
        
        return reward_responses, punishment_responses
    
    def classify_behavior(self, agent_pos, reward_pos, punishment_pos, movement):
        """Classify current behavior pattern"""
        try:
            reward_dist = np.linalg.norm(np.array(agent_pos) - np.array(reward_pos))
            punishment_dist = np.linalg.norm(np.array(agent_pos) - np.array(punishment_pos))
            movement_mag = np.linalg.norm(np.array(movement))
            
            if movement_mag < 0.1:
                return 'stationary'
            elif reward_dist > punishment_dist:
                return 'avoidance'
            else:
                return 'approach'
        except Exception as e:
            print(f"Error in classify_behavior: {e}")
            return 'unknown'
    
    def log_behavior(self, timestep, agent_pos, reward_pos, punishment_pos, movement, network_activation):
        """Log behavioral and network state data"""
        try:
            behavior = self.classify_behavior(agent_pos, reward_pos, punishment_pos, movement)
            
            self.behavior_log.append({
                'timestep': timestep,
                'behavior': behavior,
                'agent_pos': np.array(agent_pos),
                'reward_pos': reward_pos,
                'punishment_pos': punishment_pos,
                'movement': np.array(movement)
            })
            
            self.network_states.append(network_activation)
            self.movement_log.append(np.array(movement))
            
        except Exception as e:
            print(f"Error in log_behavior: {e}")
    
    def generate_analysis_report(self, generation):
        """Generate analysis report with visualizations"""
        try:
            # Ensure we have data to analyze
            if not self.behavior_log:
                return {
                    'generation': generation,
                    'behavior_distribution': {'approach': 0, 'avoidance': 0, 'stationary': 0},
                    'average_speed': 0.0,
                    'average_activation': np.zeros(2)  # Default for 2 outputs
                }
            
            # Behavior analysis
            behaviors = [log['behavior'] for log in self.behavior_log]
            behavior_counts = {
                'approach': behaviors.count('approach'),
                'avoidance': behaviors.count('avoidance'),
                'stationary': behaviors.count('stationary')
            }
            
            # Movement analysis - calculate speed from movement vectors
            movements = np.array(self.movement_log)
            if len(movements) > 0:
                speeds = np.linalg.norm(movements, axis=1)
                avg_speed = np.mean(speeds)
            else:
                avg_speed = 0.0
            
            # Network activation analysis
            activation_patterns = np.array(self.network_states) if self.network_states else np.zeros((1, 2))
            avg_activation = np.mean(activation_patterns, axis=0)
            
            # Reset logs for next generation
            self.reset_logs()
            
            return {
                'generation': generation,
                'behavior_distribution': behavior_counts,
                'average_speed': float(avg_speed),
                'average_activation': avg_activation.tolist()
            }
            
        except Exception as e:
            print(f"Error in generate_analysis_report: {e}")
            return {
                'generation': generation,
                'behavior_distribution': {'approach': 0, 'avoidance': 0, 'stationary': 0},
                'average_speed': 0.0,
                'average_activation': [0.0, 0.0]
            }