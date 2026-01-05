
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
