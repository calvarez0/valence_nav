import pygame
import numpy as np
import cv2
from neat import nn

def create_generation_video(genome, config, env, filename):

    pygame.init()
    pygame.font.init()

    def get_gradient_color(reward_intensity, punishment_intensity):
        """Convert reward/punishment intensities to RGB color"""
        # Normalize intensities to -1 to 1 range
        net_intensity = np.clip(reward_intensity - punishment_intensity, -1, 1)
        
        if net_intensity > 0:
            # Positive (reward) - shift from white to blue
            blue = int(255 * net_intensity)
            return (255 - blue//2, 255 - blue//2, 255)
        else:
            # Negative (punishment) - shift from white to red
            red = int(255 * abs(net_intensity))
            return (255, 255 - red//2, 255 - red//2)


    scale = 8
    width = env.size * scale
    height = env.size * scale
    surface = pygame.Surface((width, height))
    
    net = nn.FeedForwardNetwork.create(genome, config)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
    
    agent_pos = np.array([env.size//2, env.size//2])  # Start in middle
    steps = 0
    max_steps = 500
    trajectory = []
    rewards_collected = 0
    
    while steps < max_steps:
        surface.fill((255, 255, 255))
        
        # Draw sensory gradients
        gradient_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        for x in range(0, width, scale):
            for y in range(0, height, scale):
                pos = np.array([x/scale, y/scale])
                
                # Calculate reward and punishment intensities
                reward_intensity = 0
                if env.active_reward:
                    r_dist = np.linalg.norm(pos - np.array(env.active_reward['pos']))
                    if r_dist < env.max_sensory_distance:
                        reward_intensity = np.exp(-r_dist / env.max_sensory_distance)
                
                punishment_intensity = 0
                for punct in env.points['static_punishments']:
                    p_dist = np.linalg.norm(pos - np.array(punct['pos']))
                    if p_dist < punct['radius'] * 2:
                        punishment_intensity += np.exp(-p_dist / (punct['radius'] * 2))
                
                # Only draw if there's any intensity
                if reward_intensity > 0 or punishment_intensity > 0:
                    color = get_gradient_color(reward_intensity, punishment_intensity)
                    alpha = int(255 * max(reward_intensity, punishment_intensity))
                    pygame.draw.rect(gradient_surface, (*color, alpha//2),
                                  (x, y, scale, scale))
        
        surface.blit(gradient_surface, (0, 0))
        
        # Draw active reward
        if env.active_reward:
            pos = (int(env.active_reward['pos'][0] * scale), 
                  int(env.active_reward['pos'][1] * scale))
            pygame.draw.circle(surface, (0, 0, 255), pos, 3)
        
        # Draw punishments
        for punct in env.points['static_punishments']:
            pos = (int(punct['pos'][0] * scale), int(punct['pos'][1] * scale))
            pygame.draw.circle(surface, (255, 0, 0), pos, 3)
        
        # Draw agent
        agent_pixel_pos = (int(agent_pos[0] * scale), int(agent_pos[1] * scale))
        pygame.draw.circle(surface, (0, 255, 0), agent_pixel_pos, scale // 2)
        
        # Draw trajectory
        if len(trajectory) > 1:
            scaled_trajectory = [(int(x * scale), int(y * scale)) for x, y in trajectory]
            pygame.draw.lines(surface, (0, 200, 0), False, scaled_trajectory, 1)
        
        # Get neural network output and move agent
        inputs = env.get_sensory_input(agent_pos)
        output = net.activate(inputs)
        
        dx = np.clip(output[0], -1, 1)
        dy = np.clip(output[1], -1, 1)
        movement = np.array([dx, dy])
        
        new_pos = np.clip(agent_pos + movement * 2, 0, env.size-1)
        trajectory.append(agent_pos.copy())
        
        # Update environment and check for reward collection
        reward = env.update(new_pos)
        if reward > 0:
            rewards_collected += 1
        
        agent_pos = new_pos
        
        # Add reward counter to frame
        font = pygame.font.Font(None, 36)
        text = font.render(f'Rewards: {rewards_collected}', True, (0, 0, 0))
        surface.blit(text, (10, 10))
        
        # Convert and save frame
        frame = pygame.surfarray.array3d(surface)
        frame = frame.swapaxes(0, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
        
        steps += 1
    
    video.release()