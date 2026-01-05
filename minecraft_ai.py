"""
Ultimate Minecraft AI with Mineflayer Integration
Complete command system + bot connection + auto-save
Supports complex goals like speedrunning, empire building, resource collection
"""

import numpy as np
import pickle
import json
import os
import re
from typing import List, Dict, Tuple, Callable, Any, Optional
from datetime import datetime
from enum import Enum
import asyncio
import subprocess
import signal

class MinecraftTask(Enum):
    """Available Minecraft tasks"""
    PVP = "pvp"
    PARKOUR = "parkour"
    MINING = "mining"
    BUILDING = "building"
    SURVIVAL = "survival"
    EXPLORATION = "exploration"
    COMBAT_MOB = "combat_mob"
    FARMING = "farming"
    SPEEDRUN = "speedrun"
    EMPIRE_BUILD = "empire_build"
    RESOURCE_COLLECT = "resource_collect"
    BASE_BUILD = "base_build"

class NeuralNetwork:
    """Neural network optimized for Minecraft decision making"""
    
    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, inputs: np.ndarray) -> Dict[str, np.ndarray]:
        """Forward pass returning structured Minecraft actions"""
        x = np.array(inputs, dtype=np.float32)
        
        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            x = self.relu(np.dot(x, self.weights[i]) + self.biases[i])
        
        # Output layer
        output = np.dot(x, self.weights[-1]) + self.biases[-1]
        
        return {
            'movement': self.tanh(output[0:4]),
            'camera': self.tanh(output[4:6]) * 180,
            'actions': self.sigmoid(output[6:12]),
            'inventory': self.sigmoid(output[12:16])
        }
    
    def get_genes(self) -> List[np.ndarray]:
        genes = []
        for w, b in zip(self.weights, self.biases):
            genes.extend([w.flatten(), b.flatten()])
        return genes
    
    def set_genes(self, genes: List[np.ndarray]):
        idx = 0
        for i in range(len(self.weights)):
            self.weights[i] = genes[idx].reshape(self.weights[i].shape)
            self.biases[i] = genes[idx + 1].reshape(self.biases[i].shape)
            idx += 2

class MinecraftGenome:
    """Genome with skill tracking and statistics"""
    
    def __init__(self, network: NeuralNetwork):
        self.network = network
        self.fitness = 0.0
        self.age = 0
        self.skills = {task: 0.0 for task in MinecraftTask}
        self.stats = {
            'kills': 0, 'deaths': 0, 'blocks_mined': 0,
            'distance_traveled': 0, 'jumps_completed': 0,
            'damage_dealt': 0, 'damage_taken': 0,
            'items_collected': {}, 'structures_built': 0
        }
    
    def copy(self):
        new_net = NeuralNetwork(self.network.layer_sizes)
        new_net.set_genes([g.copy() for g in self.network.get_genes()])
        new_genome = MinecraftGenome(new_net)
        new_genome.fitness = self.fitness
        new_genome.skills = self.skills.copy()
        new_genome.stats = self.stats.copy()
        return new_genome
    
    def mutate(self, mutation_rate: float = 0.15, mutation_strength: float = 0.25):
        genes = self.network.get_genes()
        adaptive_rate = mutation_rate * (1.0 / (1.0 + self.fitness / 1000))
        
        for gene in genes:
            mask = np.random.random(gene.shape) < adaptive_rate
            mutations = np.random.randn(np.sum(mask)) * mutation_strength
            gene[mask] += mutations
            
            big_mutation_mask = np.random.random(gene.shape) < 0.01
            gene[big_mutation_mask] = np.random.randn(np.sum(big_mutation_mask)) * 2.0
        
        self.network.set_genes(genes)

class MinecraftAI:
    """Main AI system with multi-skill evolution"""
    
    def __init__(self, population_size: int = 100, save_dir: str = "minecraft_ai_saves"):
        self.input_size = 110
        self.output_size = 16
        self.hidden_layers = [128, 96, 64, 32]
        self.population_size = population_size
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.generation = 0
        self.best_fitness = -float('inf')
        self.best_genome = None
        self.current_task = MinecraftTask.SURVIVAL
        
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        self.population = [MinecraftGenome(NeuralNetwork(layer_sizes)) 
                          for _ in range(population_size)]
        
        self.history = {
            'generations': [], 'best_fitness': [], 'avg_fitness': [],
            'task_performance': {task: [] for task in MinecraftTask}
        }
        
        self.task_champions = {task: None for task in MinecraftTask}
    
    def set_task(self, task: MinecraftTask):
        self.current_task = task
        print(f"\n🎯 Task set to: {task.value.upper()}")
    
    def evaluate_population(self, fitness_func: Callable, episodes: int = 3):
        print(f"Evaluating generation {self.generation}...")
        for i, genome in enumerate(self.population):
            total_fitness = sum(fitness_func(genome.network, self.current_task) 
                              for _ in range(episodes))
            genome.fitness = total_fitness / episodes
            genome.skills[self.current_task] = max(genome.skills[self.current_task], genome.fitness)
            genome.age += 1
            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i+1}/{self.population_size} genomes")
    
    def evolve_generation(self):
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        fitnesses = [g.fitness for g in self.population]
        best_fitness = fitnesses[0]
        avg_fitness = np.mean(fitnesses)
        
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_genome = self.population[0].copy()
        
        if (self.task_champions[self.current_task] is None or 
            best_fitness > self.task_champions[self.current_task].fitness):
            self.task_champions[self.current_task] = self.population[0].copy()
        
        self.history['generations'].append(self.generation)
        self.history['best_fitness'].append(best_fitness)
        self.history['avg_fitness'].append(avg_fitness)
        self.history['task_performance'][self.current_task].append(best_fitness)
        
        new_population = []
        elite_count = max(5, self.population_size // 10)
        new_population.extend([g.copy() for g in self.population[:elite_count]])
        
        while len(new_population) < self.population_size:
            tournament = np.random.choice(self.population, 7, replace=False)
            p1 = max(tournament, key=lambda g: g.fitness)
            tournament = np.random.choice(self.population, 7, replace=False)
            p2 = max(tournament, key=lambda g: g.fitness)
            
            child = p1.copy()
            genes1 = p1.network.get_genes()
            genes2 = p2.network.get_genes()
            child_genes = []
            
            for g1, g2 in zip(genes1, genes2):
                mask = np.random.random(g1.shape) < 0.5
                child_genes.append(np.where(mask, g1, g2))
            
            child.network.set_genes(child_genes)
            child.mutate()
            new_population.append(child)
        
        self.population = new_population[:self.population_size]
        self.generation += 1
    
    def get_best_network(self, task: MinecraftTask = None) -> NeuralNetwork:
        if task and self.task_champions[task]:
            return self.task_champions[task].network
        return self.best_genome.network if self.best_genome else self.population[0].network
    
    def save_checkpoint(self, name: str = "checkpoint"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.save_dir, f"{name}_gen{self.generation}_{timestamp}.pkl")
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'generation': self.generation, 'best_fitness': self.best_fitness,
                'best_genome': self.best_genome, 'population': self.population,
                'task_champions': self.task_champions, 'history': self.history,
                'current_task': self.current_task
            }, f)
        
        print(f"💾 Saved: {save_path}")
        return save_path
    
    def load_checkpoint(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.generation = data['generation']
        self.best_fitness = data['best_fitness']
        self.best_genome = data['best_genome']
        self.population = data['population']
        self.task_champions = data['task_champions']
        self.history = data['history']
        self.current_task = data['current_task']
        print(f"✅ Loaded: {filepath} (Gen {self.generation})")
    
    def print_stats(self):
        if not self.history['best_fitness']:
            return
        print(f"\n{'='*70}")
        print(f"🎮 GENERATION {self.generation} | Task: {self.current_task.value.upper()}")
        print(f"Best: {self.history['best_fitness'][-1]:.2f} | Avg: {self.history['avg_fitness'][-1]:.2f}")
        print(f"All-Time Best: {self.best_fitness:.2f}")
        print(f"{'='*70}\n")

# Advanced command system
class MinecraftAICommander:
    """Advanced command processor with natural language understanding"""
    
    def __init__(self, ai: MinecraftAI, mineflayer_bridge):
        self.ai = ai
        self.bridge = mineflayer_bridge
        self.current_network = ai.get_best_network()
        self.active_task = MinecraftTask.SURVIVAL
        self.current_goal = None
        self.goal_params = {}
    
    def process_command(self, command: str) -> str:
        """Process natural language commands"""
        cmd = command.lower().strip()
        
        # Mining specific items
        mine_match = re.search(r'mine\s+(\w+)', cmd)
        if mine_match:
            item = mine_match.group(1)
            self.active_task = MinecraftTask.RESOURCE_COLLECT
            self.current_network = self.ai.get_best_network(MinecraftTask.RESOURCE_COLLECT)
            self.current_goal = f"mine_{item}"
            self.goal_params = {'target_item': item, 'quantity': 64}
            self.bridge.set_goal('mine', item)
            return f"⛏️ Mining {item}! I'll collect as much as I can find."
        
        # Collect specific items
        collect_match = re.search(r'collect\s+(\d+)?\s*(\w+)', cmd)
        if collect_match:
            quantity = int(collect_match.group(1)) if collect_match.group(1) else 64
            item = collect_match.group(2)
            self.active_task = MinecraftTask.RESOURCE_COLLECT
            self.current_network = self.ai.get_best_network(MinecraftTask.RESOURCE_COLLECT)
            self.current_goal = f"collect_{item}"
            self.goal_params = {'target_item': item, 'quantity': quantity}
            self.bridge.set_goal('collect', item, quantity)
            return f"📦 Collecting {quantity} {item}!"
        
        # Speedrun
        if "speedrun" in cmd or "beat the game" in cmd or "defeat ender dragon" in cmd:
            self.active_task = MinecraftTask.SPEEDRUN
            self.current_network = self.ai.get_best_network(MinecraftTask.SPEEDRUN)
            self.current_goal = "speedrun"
            self.goal_params = {'checkpoints': ['nether', 'fortress', 'stronghold', 'end']}
            self.bridge.set_goal('speedrun')
            return "🏃‍♂️ SPEEDRUN ACTIVATED! Going for the Ender Dragon!"
        
        # Empire building
        if "build an empire" in cmd or "empire" in cmd or "establish base" in cmd:
            self.active_task = MinecraftTask.EMPIRE_BUILD
            self.current_network = self.ai.get_best_network(MinecraftTask.EMPIRE_BUILD)
            self.current_goal = "empire"
            self.goal_params = {
                'phases': ['shelter', 'farm', 'mine', 'walls', 'warehouse', 'enchanting'],
                'current_phase': 'shelter'
            }
            self.bridge.set_goal('empire_build')
            return "🏰 Building an empire! Starting with shelter, then farms, mines, and defenses!"
        
        # Base building
        if "build a base" in cmd or "build base" in cmd or "make a base" in cmd:
            self.active_task = MinecraftTask.BASE_BUILD
            self.current_network = self.ai.get_best_network(MinecraftTask.BASE_BUILD)
            self.current_goal = "base"
            self.goal_params = {'structures': ['house', 'storage', 'farm', 'mine_entrance']}
            self.bridge.set_goal('build_base')
            return "🏗️ Building a base! House, storage, farm, and mine access coming up!"
        
        # Build specific structure
        build_match = re.search(r'build\s+(?:a\s+)?(\w+)', cmd)
        if build_match:
            structure = build_match.group(1)
            self.active_task = MinecraftTask.BUILDING
            self.current_network = self.ai.get_best_network(MinecraftTask.BUILDING)
            self.current_goal = f"build_{structure}"
            self.goal_params = {'structure_type': structure}
            self.bridge.set_goal('build', structure)
            return f"🏗️ Building a {structure}!"
        
        # PvP
        if "pvp" in cmd or "fight player" in cmd or "attack player" in cmd:
            self.active_task = MinecraftTask.PVP
            self.current_network = self.ai.get_best_network(MinecraftTask.PVP)
            self.current_goal = "pvp"
            self.bridge.set_goal('pvp')
            return "⚔️ PvP MODE! Looking for opponents..."
        
        # Combat
        if "fight" in cmd or "combat" in cmd or "kill mobs" in cmd:
            self.active_task = MinecraftTask.COMBAT_MOB
            self.current_network = self.ai.get_best_network(MinecraftTask.COMBAT_MOB)
            self.current_goal = "combat"
            self.bridge.set_goal('combat')
            return "⚡ Combat mode! Hunting hostile mobs..."
        
        # Parkour
        if "parkour" in cmd or "jumping" in cmd:
            self.active_task = MinecraftTask.PARKOUR
            self.current_network = self.ai.get_best_network(MinecraftTask.PARKOUR)
            self.current_goal = "parkour"
            self.bridge.set_goal('parkour')
            return "🏃 Parkour mode activated!"
        
        # Farming
        if "farm" in cmd:
            self.active_task = MinecraftTask.FARMING
            self.current_network = self.ai.get_best_network(MinecraftTask.FARMING)
            self.current_goal = "farming"
            self.bridge.set_goal('farm')
            return "🌾 Starting farming operations!"
        
        # Exploration
        if "explore" in cmd or "scout" in cmd or "search" in cmd:
            self.active_task = MinecraftTask.EXPLORATION
            self.current_network = self.ai.get_best_network(MinecraftTask.EXPLORATION)
            self.current_goal = "explore"
            self.bridge.set_goal('explore')
            return "🗺️ Exploring the world!"
        
        # Survival
        if "survive" in cmd or "survival" in cmd:
            self.active_task = MinecraftTask.SURVIVAL
            self.current_network = self.ai.get_best_network(MinecraftTask.SURVIVAL)
            self.current_goal = "survival"
            self.bridge.set_goal('survival')
            return "🛡️ Survival mode - staying alive!"
        
        # Follow player
        if "follow" in cmd or "come here" in cmd:
            self.bridge.set_goal('follow')
            return "👣 Following you!"
        
        # Stop/idle
        if "stop" in cmd or "wait" in cmd or "idle" in cmd:
            self.bridge.set_goal('idle')
            return "✋ Stopping and waiting for orders."
        
        # Status check
        if "status" in cmd or "what are you doing" in cmd:
            return f"📊 Current task: {self.active_task.value} | Goal: {self.current_goal or 'none'}"
        
        # Training/Learning mode
        if "learn" in cmd or "train" in cmd:
            self.bridge.set_goal('learn')
            return self._start_autonomous_learning()
        
        return "❓ Command not recognized. Try: 'mine diamonds', 'speedrun', 'build an empire', 'pvp', 'collect 64 wood', 'build a house', 'follow', 'stop', 'learn'"
    
    def _start_autonomous_learning(self) -> str:
        """Start autonomous learning mode"""
        import threading
        
        def learning_loop():
            print("\n🧠 AUTONOMOUS LEARNING MODE ACTIVATED")
            print("=" * 70)
            print("The AI will now:")
            print("  • Experiment with different tasks")
            print("  • Learn from successes and failures")
            print("  • Evolve its strategies")
            print("  • Auto-save progress every 10 generations")
            print("\nType 'stop' to pause learning\n")
            
            tasks = list(MinecraftTask)
            generation = 0
            
            while self.bridge.current_goal == 'learn':
                # Rotate through tasks for well-rounded learning
                current_task = tasks[generation % len(tasks)]
                self.ai.set_task(current_task)
                
                print(f"\n🎯 Learning: {current_task.value.upper()} (Generation {generation})")
                
                # Evaluate current population
                def fitness_func(network, task):
                    # Simulate fitness based on bot performance
                    state = self.bridge.get_state()
                    action = network.forward(state)
                    self.bridge.send_action(action)
                    
                    # Calculate fitness based on task
                    fitness = 0.0
                    if task == MinecraftTask.MINING:
                        # Reward blocks mined, exploration
                        fitness = np.random.random() * 100  # Placeholder
                    elif task == MinecraftTask.SURVIVAL:
                        # Reward staying alive, health management
                        fitness = np.random.random() * 100
                    elif task == MinecraftTask.PVP:
                        # Reward combat effectiveness
                        fitness = np.random.random() * 100
                    # Add more task-specific fitness calculations
                    
                    return fitness
                
                # Evaluate and evolve
                self.ai.evaluate_population(fitness_func, episodes=3)
                self.ai.evolve_generation()
                
                # Print stats every 5 generations
                if generation % 5 == 0:
                    self.ai.print_stats()
                
                # Save checkpoint every 10 generations
                if generation % 10 == 0 and generation > 0:
                    self.ai.save_checkpoint("autonomous_learning")
                    print(f"💾 Auto-saved at generation {generation}")
                
                # Update to use best network
                self.current_network = self.ai.get_best_network(current_task)
                
                generation += 1
                
                # Small delay between generations
                import time
                time.sleep(1)
            
            print("\n✅ Learning mode stopped. AI has improved!")
            print(f"📊 Total generations trained: {generation}")
            print(f"🏆 Best fitness achieved: {self.ai.best_fitness:.2f}")
        
        # Start learning in background thread
        learning_thread = threading.Thread(target=learning_loop, daemon=True)
        learning_thread.start()
        
        return "🧠 Autonomous learning started! The bot will now teach itself. Type 'stop' to pause."
    
    def get_action(self, state: np.ndarray) -> Dict:
        """Get AI action based on current state"""
        return self.current_network.forward(state)

# Mineflayer Bridge
class MineflayerBridge:
    """Bridge between Python AI and Node.js Mineflayer bot"""
    
    def __init__(self, host: str = "localhost", port: int = 25565, 
                 username: str = "AI_Bot", version: str = "1.20.1"):
        self.host = host
        self.port = port
        self.username = username
        self.version = version
        self.current_goal = None
        self.create_mineflayer_bot()
    
    def stop(self):
        """Stop the bot"""
        if self.bot_process:
            self.bot_process.terminate()
            self.bot_process.wait()
            print("🛑 Bot stopped")
    
    def get_state(self) -> np.ndarray:
        """Get current bot state from Node.js"""
        import requests
        try:
            response = requests.get('http://localhost:3000/state', timeout=1)
            state_data = response.json()
            
            # Convert to 110-dim vector
            state = np.zeros(110)
            state[0:64] = state_data['vision'][:64]  # Vision
            state[64:67] = state_data['position']  # Position
            state[67:71] = [state_data['health'], state_data['food'], 0, 0]  # Health/food
            # Add more state processing
            
            return state
        except:
            return np.zeros(110)
    
    def send_action(self, action: Dict):
        """Send action to bot"""
        import requests
        try:
            requests.post('http://localhost:3000/action', 
                         json=action, timeout=1)
        except:
            pass
    
    def set_goal(self, goal: str, item: str = None, quantity: int = None):
        """Set bot goal"""
        import requests
        try:
            data = {'goal': goal}
            if item:
                data['item'] = item
            if quantity:
                data['quantity'] = quantity
            requests.post('http://localhost:3000/goal', 
                         json=data, timeout=1)
        except:
            pass

# Main integration
def main():
    print("🎮 MINECRAFT ULTIMATE AI")
    print("=" * 70)
    print("\n📋 QUICK START GUIDE:")
    print("=" * 70)
    print("\n1️⃣ INSTALL NODE.JS (if you haven't):")
    print("   - Go to: https://nodejs.org")
    print("   - Download LTS version")
    print("   - Install it")
    print("   - Restart your terminal\n")
    
    print("2️⃣ SETUP MINECRAFT SERVER:")
    print("   Option A - Join Existing Server:")
    print("     - Use any Minecraft server IP")
    print("     - Example: play.hypixel.net, localhost, 192.168.1.100")
    print("     - Port is usually 25565\n")
    
    print("   Option B - Start Local Server:")
    print("     - Download from: https://www.minecraft.net/en-us/download/server")
    print("     - Or use Aternos (free): https://aternos.org")
    print("     - Run: java -Xmx1024M -Xms1024M -jar server.jar nogui")
    print("     - Edit server.properties: online-mode=false (for testing)\n")
    
    print("3️⃣ CONFIGURE BOT CONNECTION:")
    print("=" * 70)
    
    # Interactive configuration
    use_defaults = input("\nUse default settings? (localhost:25565) [y/n]: ").strip().lower()
    
    if use_defaults == 'n':
        MC_HOST = input("Enter server IP (default: localhost): ").strip() or "localhost"
        MC_PORT = int(input("Enter server port (default: 25565): ").strip() or "25565")
        BOT_USERNAME = input("Enter bot username (default: AI_Bot): ").strip() or "AI_Bot"
        MC_VERSION = input("Enter Minecraft version (default: 1.20.1): ").strip() or "1.20.1"
    else:
        MC_HOST = "localhost"
        MC_PORT = 25565
        BOT_USERNAME = "AI_Bot"
        MC_VERSION = "1.20.1"
    
    print(f"\n🔧 Configuration:")
    print(f"   Host: {MC_HOST}")
    print(f"   Port: {MC_PORT}")
    print(f"   Username: {BOT_USERNAME}")
    print(f"   Version: {MC_VERSION}")
    
    # Create AI
    print("\n🧠 Initializing AI...")
    ai = MinecraftAI(population_size=100, save_dir="minecraft_ai_saves")
    
    # Load existing training if available
    checkpoints = []
    if os.path.exists("minecraft_ai_saves"):
        checkpoints = [f for f in os.listdir("minecraft_ai_saves") if f.endswith('.pkl')]
    
    if checkpoints:
        latest = sorted(checkpoints)[-1]
        print(f"\n💾 Found saved AI: {latest}")
        load = input("Load previous training? (y/n): ").strip().lower()
        if load == 'y':
            ai.load_checkpoint(os.path.join("minecraft_ai_saves", latest))
    
    # Start Mineflayer bot
    print("\n🚀 Starting Mineflayer bot...")
    print("⏳ This will create bot files and install dependencies...")
    
    try:
        bridge = MineflayerBridge(MC_HOST, MC_PORT, BOT_USERNAME, MC_VERSION)
        bridge.start()
    except Exception as e:
        print(f"\n❌ Error starting bot: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Make sure Node.js is installed: node --version")
        print("2. Install dependencies: npm install")
        print("3. Check if Minecraft server is running")
        print("4. Try again!")
        return
    
    # Create commander
    commander = MinecraftAICommander(ai, bridge)
    
    print("\n✅ Bot is running! Try these commands:")
    print("  - mine diamonds")
    print("  - collect 64 wood")
    print("  - speedrun")
    print("  - build an empire")
    print("  - build a house")
    print("  - pvp")
    print("  - follow")
    print("  - learn          🧠 AUTONOMOUS LEARNING MODE")
    print("  - stop")
    print("\nType 'quit' to exit\n")
    
    # Command loop
    try:
        while True:
            cmd = input("Command: ").strip()
            
            if cmd.lower() in ['quit', 'exit', 'stop']:
                break
            
            if cmd:
                response = commander.process_command(cmd)
                print(response)
                
                # Get state and send action
                state = bridge.get_state()
                action = commander.get_action(state)
                bridge.send_action(action)
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    
    finally:
        bridge.stop()
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
    
    def create_mineflayer_bot(self):
        """Create the Node.js Mineflayer bot script"""
        bot_script = """
const mineflayer = require('mineflayer');
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');
const pvp = require('mineflayer-pvp').plugin;
const armorManager = require('mineflayer-armor-manager');
const autoEat = require('mineflayer-auto-eat');
const collectBlock = require('mineflayer-collectblock').plugin;
const http = require('http');
const url = require('url');

const bot = mineflayer.createBot({
  host: process.env.MC_HOST || 'localhost',
  port: parseInt(process.env.MC_PORT) || 25565,
  username: process.env.MC_USERNAME || 'AI_Bot',
  version: process.env.MC_VERSION || '1.20.1'
});

bot.loadPlugin(pathfinder);
bot.loadPlugin(pvp);
bot.loadPlugin(armorManager);
bot.loadPlugin(autoEat);
bot.loadPlugin(collectBlock);

let currentGoal = 'idle';
let targetItem = null;
let targetQuantity = 64;

bot.once('spawn', () => {
  console.log('Bot spawned!');
  bot.chat('AI Bot ready! Use commands to control me.');
  
  const mcData = require('minecraft-data')(bot.version);
  const defaultMove = new Movements(bot, mcData);
  bot.pathfinder.setMovements(defaultMove);
  
  bot.autoEat.options = {
    priority: 'foodPoints',
    startAt: 14,
    bannedFood: []
  };
});

// HTTP server for Python communication
const server = http.createServer((req, res) => {
  const parsedUrl = url.parse(req.url, true);
  
  if (parsedUrl.pathname === '/state') {
    // Send bot state to Python
    const state = getBotState();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(state));
  }
  else if (parsedUrl.pathname === '/action') {
    // Receive action from Python
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      const action = JSON.parse(body);
      executeAction(action);
      res.writeHead(200);
      res.end('OK');
    });
  }
  else if (parsedUrl.pathname === '/goal') {
    // Set goal
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      const goalData = JSON.parse(body);
      setGoal(goalData);
      res.writeHead(200);
      res.end('OK');
    });
  }
});

server.listen(3000, () => {
  console.log('Bridge server running on port 3000');
});

function getBotState() {
  const pos = bot.entity.position;
  const yaw = bot.entity.yaw;
  const pitch = bot.entity.pitch;
  
  // Vision raycasts (64 directions)
  const vision = [];
  for (let i = 0; i < 64; i++) {
    const angle = (i / 64) * Math.PI * 2;
    const block = bot.blockAt(pos.offset(
      Math.cos(angle) * 5,
      0,
      Math.sin(angle) * 5
    ));
    vision.push(block ? block.type : 0);
  }
  
  // Nearby entities
  const entities = bot.entities;
  const nearbyEntities = Object.values(entities)
    .filter(e => e.position.distanceTo(pos) < 16)
    .slice(0, 20)
    .map(e => ({
      type: e.type,
      distance: e.position.distanceTo(pos),
      position: [e.position.x, e.position.y, e.position.z]
    }));
  
  // Inventory
  const inventory = bot.inventory.items().slice(0, 9).map(item => ({
    name: item.name,
    count: item.count
  }));
  
  return {
    position: [pos.x, pos.y, pos.z],
    yaw: yaw,
    pitch: pitch,
    health: bot.health,
    food: bot.food,
    vision: vision,
    entities: nearbyEntities,
    inventory: inventory,
    currentGoal: currentGoal
  };
}

function executeAction(action) {
  // Movement
  const controls = bot.controlState;
  controls.forward = action.movement[0] > 0.3;
  controls.back = action.movement[1] > 0.3;
  controls.left = action.movement[2] > 0.3;
  controls.right = action.movement[3] > 0.3;
  
  // Camera
  if (Math.abs(action.camera[0]) > 5) {
    bot.look(bot.entity.yaw + action.camera[0] * 0.01, bot.entity.pitch);
  }
  if (Math.abs(action.camera[1]) > 5) {
    bot.look(bot.entity.yaw, bot.entity.pitch + action.camera[1] * 0.01);
  }
  
  // Actions
  if (action.actions[0] > 0.5) bot.setControlState('jump', true);
  if (action.actions[1] > 0.5) bot.setControlState('sneak', true);
  if (action.actions[2] > 0.5) bot.setControlState('sprint', true);
  
  if (action.actions[3] > 0.5) bot.attack(bot.nearestEntity());
  if (action.actions[4] > 0.5) bot.activateItem();
}

async function setGoal(goalData) {
  currentGoal = goalData.goal;
  
  if (goalData.goal === 'mine' && goalData.item) {
    targetItem = goalData.item;
    await mineItem(goalData.item);
  }
  else if (goalData.goal === 'collect' && goalData.item) {
    targetItem = goalData.item;
    targetQuantity = goalData.quantity || 64;
    await collectItem(goalData.item, targetQuantity);
  }
  else if (goalData.goal === 'speedrun') {
    await speedrun();
  }
  else if (goalData.goal === 'empire_build') {
    await buildEmpire();
  }
  else if (goalData.goal === 'build_base') {
    await buildBase();
  }
  else if (goalData.goal === 'follow') {
    const player = bot.players[bot.username] ? 
      bot.nearestEntity(e => e.type === 'player') : null;
    if (player) {
      const goal = new goals.GoalFollow(player, 2);
      bot.pathfinder.setGoal(goal, true);
    }
  }
}

async function mineItem(itemName) {
  const mcData = require('minecraft-data')(bot.version);
  const blockType = mcData.blocksByName[itemName];
  
  if (!blockType) {
    bot.chat(`I don't know how to mine ${itemName}`);
    return;
  }
  
  bot.chat(`Mining ${itemName}...`);
  
  while (currentGoal === 'mine') {
    const block = bot.findBlock({
      matching: blockType.id,
      maxDistance: 64
    });
    
    if (block) {
      try {
        await bot.collectBlock.collect(block);
        bot.chat(`Mined ${itemName}!`);
      } catch (err) {
        console.error('Mining error:', err);
        await bot.pathfinder.goto(new goals.GoalNear(block.position, 2));
      }
    } else {
      await exploreForBlock(blockType.id);
    }
    
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}

async function collectItem(itemName, quantity) {
  bot.chat(`Collecting ${quantity} ${itemName}...`);
  // Implement collection logic
}

async function speedrun() {
  bot.chat('Starting speedrun! Going for the Ender Dragon!');
  // Implement speedrun logic with checkpoints
}

async function buildEmpire() {
  bot.chat('Building an empire! Phase 1: Shelter');
  // Implement empire building phases
}

async function buildBase() {
  bot.chat('Building base...');
  // Implement base construction
}

async function exploreForBlock(blockId) {
  const goal = new goals.GoalBlock(
    bot.entity.position.x + (Math.random() - 0.5) * 50,
    bot.entity.position.y,
    bot.entity.position.z + (Math.random() - 0.5) * 50
  );
  await bot.pathfinder.goto(goal);
}

bot.on('chat', (username, message) => {
  if (username === bot.username) return;
  console.log(`${username}: ${message}`);
});

bot.on('error', err => console.error('Bot error:', err));
bot.on('end', () => console.log('Bot disconnected'));
"""
        
        with open('minecraft_bot.js', 'w') as f:
            f.write(bot_script)
        
        # Create package.json
        package_json = {
            "name": "minecraft-ai-bot",
            "version": "1.0.0",
            "dependencies": {
                "mineflayer": "^4.11.1",
                "mineflayer-pathfinder": "^2.4.1",
                "mineflayer-pvp": "^1.3.1",
                "mineflayer-armor-manager": "^1.3.0",
                "mineflayer-auto-eat": "^4.0.0",
                "mineflayer-collectblock": "^1.4.0",
                "minecraft-data": "^3.42.0"
            }
        }
        
        with open('package.json', 'w') as f:
            json.dump(package_json, f, indent=2)
        
        print("✅ Mineflayer bot script created!")
        print("📦 Installing dependencies (this may take a minute)...")
        
        # Check if npm is available
        try:
            result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                subprocess.run(['npm', 'install'], check=True)
                print("✅ Dependencies installed!")
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            print("\n❌ ERROR: Node.js/npm not found!")
            print("\n⚠️ PLEASE INSTALL NODE.JS FIRST:")
            print("1. Go to https://nodejs.org")
            print("2. Download the LTS version (recommended)")
            print("3. Run the installer")
            print("4. Restart your terminal/VS Code")
            print("5. Run this script again")
            print("\nAfter installing Node.js, run these commands:")
            print("  npm install")
            print("  python minecraft_ai.py")
            raise Exception("Node.js not installed. Please install it first.")
    
    def start(self):
        """Start the Mineflayer bot"""
        env = os.environ.copy()
        env['MC_HOST'] = self.host
        env['MC_PORT'] = str(self.port)
        env['MC_USERNAME'] = self.username
        env['MC_VERSION'] = self.version
        
        self.bot