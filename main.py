import os
import neat
import numpy as np
import networkx as nx
import time
from datetime import datetime
from network_analysis import NetworkAnalyzer
from valence_environment import ValenceEnvironment
from visualization import create_generation_video

def eval_genome(genome_id, genome, config, analyzer=None):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = ValenceEnvironment()
    
    total_reward = 0
    steps = 0
    max_steps = 500
    max_time = 15
    start_time = time.time()
    
    agent_pos = np.array([env.size//2, env.size//2])
    positions = []
    movements = []
    rewards_collected = 0
    
    # Track optimal path metrics
    optimal_path_lengths = []  # Store optimal path lengths to each reward
    actual_path_lengths = []   # Store actual path lengths taken
    total_speed = 0           # Track average movement speed
    reward_times = []         # Track time to reach each reward
    last_reward_time = 0      # Time when last reward was collected
    
    while steps < max_steps:
        if time.time() - start_time > max_time:
            print(f"Evaluation timeout for genome {genome_id}")
            break
            
        positions.append(agent_pos.copy())
        
        # Calculate optimal distance to current reward
        if env.active_reward:
            optimal_dist = np.linalg.norm(np.array(env.active_reward['pos']) - agent_pos)
        
        inputs = env.get_sensory_input(agent_pos)
        output = net.activate(inputs)
        
        dx = np.clip(output[0], -1, 1)
        dy = np.clip(output[1], -1, 1)
        movement = np.array([dx, dy])
        movements.append(movement)
        
        # Calculate movement speed
        speed = np.linalg.norm(movement)
        total_speed += speed
        
        new_pos = np.clip(agent_pos + movement * 2, 0, env.size-1)
        
        if analyzer is not None:
            rewards = env.active_reward['value'] if env.active_reward else 0
            punishments = sum(p['value'] for p in env.points['static_punishments'])
            analyzer.log_behavior(
                steps, agent_pos, rewards, punishments,
                movement, output
            )
        
        # Update position and get reward
        reward = env.update(new_pos)
        
        # If reward collected, update metrics
        if reward > 0:
            rewards_collected += 1
            current_time = steps - last_reward_time
            reward_times.append(current_time)
            last_reward_time = steps
            
            # Calculate path efficiency for this reward
            if len(positions) > 1:
                actual_path = np.sum([np.linalg.norm(positions[i+1] - positions[i]) 
                                    for i in range(len(positions)-1)])
                optimal_path_lengths.append(optimal_dist)
                actual_path_lengths.append(actual_path)
        
        total_reward += reward
        agent_pos = new_pos
        
        steps += 1
        if total_reward < -50:  # Early termination for very poor performance
            break
    
    # Calculate final fitness with emphasis on optimal path-taking
    fitness = 0
    
    # Base reward collection bonus (50 points per reward)
    fitness += rewards_collected * 50
    
    # Path efficiency component
    if len(optimal_path_lengths) > 0:
        path_efficiencies = [opt / act if act > 0 else 0 
                           for opt, act in zip(optimal_path_lengths, actual_path_lengths)]
        avg_path_efficiency = np.mean(path_efficiencies)
        fitness += avg_path_efficiency * 100  # Scale factor for path efficiency
    
    # Speed bonus (encourage faster movement)
    avg_speed = total_speed / steps if steps > 0 else 0
    fitness += avg_speed * 30  # Scale factor for speed
    
    # Time efficiency bonus (reward quick reward collection)
    if reward_times:
        avg_time_to_reward = np.mean(reward_times)
        time_bonus = max(0, (100 - avg_time_to_reward)) * 0.5
        fitness += time_bonus
    
    # Punishment avoidance component (small negative for getting too close to punishments)
    punishment_penalty = -total_reward if total_reward < 0 else 0
    fitness += punishment_penalty
    
    return fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        print(f"Evaluating genome {genome_id}...")
        genome.fitness = eval_genome(genome_id, genome, config)
        print(f"Genome {genome_id} fitness: {genome.fitness}")

def run_neat(videos_dir):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        "config.txt")
    
    pop = neat.Population(config)
    analyzer = NetworkAnalyzer()
    
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    
    class EvolutionHandler:
        def __init__(self):
            self.generation = 0
            self.best_fitness = -float('inf')
            self.checkpoint_interval = 50
            self.next_checkpoint = self.checkpoint_interval
            self.should_stop = False
            
        def evaluate_genomes(self, genomes, config):
            if self.should_stop:
                return True  # Signal to stop evolution immediately
                
            self.generation += 1
            current_best = -float('inf')
            best_genome = None
            
            # Evaluate all genomes
            for genome_id, genome in genomes:
                print(f"Evaluating genome {genome_id}...")
                genome.fitness = eval_genome(genome_id, genome, config)
                print(f"Genome {genome_id} fitness: {genome.fitness}")
                
                if genome.fitness > current_best:
                    current_best = genome.fitness
                    best_genome = genome
            
            # Update best fitness
            if current_best > self.best_fitness:
                self.best_fitness = current_best
            
            # Save champion video and data every 5 generations
            if self.generation % 5 == 0:
                # Create a subdirectory for this generation
                gen_dir = os.path.join(videos_dir, f"generation_{self.generation}")
                os.makedirs(gen_dir, exist_ok=True)
                
                # Network analysis
                network = analyzer.analyze_network(best_genome, config)
                
                # Valence processing analysis
                reward_responses, punishment_responses = analyzer.analyze_valence_processing(best_genome, config)
                
                # Behavior report
                report = analyzer.generate_analysis_report(self.generation)
                
                # Save data
                save_generation_data(self.generation, {
                    'network': network,
                    'reward_responses': reward_responses,
                    'punishment_responses': punishment_responses,
                    'behavior_report': report,
                    'fitness': self.best_fitness
                }, gen_dir)
                
                # Create and save video
                video_path = os.path.join(gen_dir, "champion.mp4")
                create_generation_video(best_genome, config, ValenceEnvironment(), video_path)
            
            print(f"\nCompleted generation {self.generation}")
            print(f"Best fitness: {self.best_fitness}")
            
            # Check if we've reached a checkpoint
            if self.generation >= self.next_checkpoint:
                while True:
                    response = input(f"\nReached checkpoint at generation {self.generation}. Continue for another {self.checkpoint_interval} generations? (y/n): ")
                    if response.lower() in ['y', 'n']:
                        if response.lower() == 'n':
                            self.should_stop = True
                            return True  # Signal to stop evolution
                        else:
                            self.next_checkpoint += self.checkpoint_interval
                            break
            
            return False  # Continue evolution
    
    # Create handler and run evolution
    handler = EvolutionHandler()
    
    # Run evolution until user chooses to stop
    winner = pop.run(handler.evaluate_genomes)
    
    return winner


def save_generation_data(generation, data, output_dir):
    """Save generation analysis data"""
    gen_dir = os.path.join(output_dir, f"gen_{generation}")
    os.makedirs(gen_dir, exist_ok=True)
    
    # Save network visualization
    nx.write_gexf(data['network'], os.path.join(gen_dir, "network.gexf"))
    
    # Save response data
    np.save(os.path.join(gen_dir, "reward_responses.npy"), data['reward_responses'])
    np.save(os.path.join(gen_dir, "punishment_responses.npy"), data['punishment_responses'])
    
    # Save behavior report
    with open(os.path.join(gen_dir, "behavior_report.txt"), 'w') as f:
        for key, value in data['behavior_report'].items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), "results", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    winner = run_neat(output_dir)
    print("\nBest genome:")
    print(winner)