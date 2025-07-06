"""
Integration of EVCAR-KG Ontology with TD3-MADDPG Agents
Enhances MARL agents' decision-making through ontological knowledge
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

class KnowledgeEnhancedDistrictAgent:
    """District agent enhanced with ontology knowledge"""
    
    def __init__(self, district_id: int, kg_integration, 
                 state_dim: int = 15, action_dim: int = 4, 
                 hidden_dim: int = 256, lr: float = 1e-4):
        
        self.district_id = district_id
        self.kg = kg_integration
        
        # Enhanced state dimension includes ontology features
        self.enhanced_state_dim = state_dim + 10  # Original + KG features
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(self.enhanced_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 4),  # 4 SoC clusters
            nn.Softmax(dim=-1)
        )
        
        # Twin critics
        self.critic1 = nn.Sequential(
            nn.Linear(self.enhanced_state_dim + action_dim * 4, 384),
            nn.ReLU(),
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Linear(384, 1)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(self.enhanced_state_dim + action_dim * 4, 384),
            nn.ReLU(),
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Linear(384, 1)
        )
        
        # Target networks
        self.actor_target = self._create_target_network(self.actor)
        self.critic1_target = self._create_target_network(self.critic1)
        self.critic2_target = self._create_target_network(self.critic2)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        
        self.update_counter = 0
        
    def _create_target_network(self, network):
        """Create a target network"""
        target = type(network)(network[0].in_features, network[-2].out_features)
        target.load_state_dict(network.state_dict())
        return target
        
    def get_observation(self, raw_obs: np.ndarray) -> np.ndarray:
        """Enhance raw observation with ontology knowledge"""
        
        # Get knowledge-enhanced observation
        kg_obs = self.kg.get_enhanced_observation('district', self.district_id)
        
        # Extract additional features from KG
        kg_features = np.array([
            1.0 if kg_obs.get('type') == 'Residential' else 0.0,
            1.0 if kg_obs.get('type') == 'Commercial' else 0.0,
            1.0 if kg_obs.get('type') == 'Industrial' else 0.0,
            1.0 if kg_obs.get('under_stress', False) else 0.0,
            1.0 if kg_obs.get('surge', False) else 0.0,
            len([n for n in kg_obs.get('neighbors', []) if n.get('outage', False)]),
            kg_obs.get('congestion', 'LOW') == 'HIGH',
            kg_obs.get('total_evs', 0) / 1000.0,  # Normalized
            kg_obs.get('critical_evs', 0) / 100.0,  # Normalized
            kg_obs.get('avg_soc', 50.0) / 100.0  # Normalized
        ])
        
        # Concatenate with raw observation
        enhanced_obs = np.concatenate([raw_obs, kg_features])
        
        return enhanced_obs
        
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Select action with ontology-based constraints"""
        
        # Get enhanced state
        enhanced_state = self.get_observation(state)
        
        # Get action from actor network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(enhanced_state).unsqueeze(0)
            action_probs = self.actor(state_tensor).squeeze(0)
            
        # Get action mask from ontology
        action_mask = self.kg.get_action_mask('district', self.district_id)
        
        # Apply mask to action probabilities
        masked_probs = action_probs.numpy()
        for i, mask in enumerate(action_mask):
            if not mask:
                masked_probs[i] = 0.0
                
        # Renormalize
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            # If all actions are masked, allow only critical charging
            masked_probs = np.zeros_like(masked_probs)
            masked_probs[0] = 1.0  # Critical SoC action
            
        # Add exploration noise if needed
        if explore:
            noise = np.random.normal(0, 0.2, size=masked_probs.shape)
            masked_probs = masked_probs + noise
            masked_probs = np.clip(masked_probs, 0, 1)
            masked_probs = masked_probs / masked_probs.sum()
            
        return masked_probs
        
    def update(self, batch: Dict, all_agents) -> Dict[str, float]:
        """Update agent with priority-weighted rewards"""
        
        states = torch.FloatTensor(batch['states'])
        actions = torch.FloatTensor(batch['actions'])
        rewards = torch.FloatTensor(batch['rewards'])
        next_states = torch.FloatTensor(batch['next_states'])
        dones = torch.FloatTensor(batch['dones'])
        
        # Calculate priority weights from ontology
        priority_weights = self._calculate_priority_weights(batch)
        
        # Weighted rewards
        weighted_rewards = rewards * torch.FloatTensor(priority_weights)
        
        # Update critics
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q1 = self.critic1_target(torch.cat([next_states, next_actions], dim=1))
            target_q2 = self.critic2_target(torch.cat([next_states, next_actions], dim=1))
            target_q = torch.min(target_q1, target_q2)
            target_value = weighted_rewards + 0.99 * target_q * (1 - dones)
            
        # Current Q values
        current_q1 = self.critic1(torch.cat([states, actions], dim=1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=1))
        
        # Critic losses
        critic1_loss = nn.MSELoss()(current_q1, target_value)
        critic2_loss = nn.MSELoss()(current_q2, target_value)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor (delayed)
        if self.update_counter % 2 == 0:
            actor_loss = -self.critic1(torch.cat([states, self.actor(states)], dim=1)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.actor, self.actor_target, tau=0.005)
            self._soft_update(self.critic1, self.critic1_target, tau=0.005)
            self._soft_update(self.critic2, self.critic2_target, tau=0.005)
            
        self.update_counter += 1
            
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item() if self.update_counter % 2 == 0 else 0
        }
        
    def _calculate_priority_weights(self, batch: Dict) -> np.ndarray:
        """Calculate priority weights based on SoC clusters"""
        weights = []
        
        for i in range(len(batch['states'])):
            # Extract SoC distribution from state
            critical_pct = batch['states'][i][-4]  # Assuming last 4 elements are SoC percentages
            low_pct = batch['states'][i][-3]
            
            # Higher weight for states with more critical/low SoC EVs
            weight = 1.0 + 0.5 * critical_pct + 0.25 * low_pct
            weights.append(weight)
            
        return np.array(weights)
        
    def _soft_update(self, source, target, tau: float = 0.005):
        """Soft update of target network"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class KnowledgeEnhancedRedistributionAgent:
    """Redistribution agent enhanced with ontology knowledge"""
    
    def __init__(self, kg_integration, num_districts: int = 6,
                 state_dim: int = 31, hidden_dim: int = 384, lr: float = 1e-4):
        
        self.kg = kg_integration
        self.num_districts = num_districts
        
        # Enhanced state includes stress indicators and district types
        self.enhanced_state_dim = state_dim + num_districts * 3  # Original + KG features per district
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(self.enhanced_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_districts * 4),  # 4 SoC clusters per district
            nn.Softmax(dim=-1)
        )
        
        # Twin critics
        self.critic1 = nn.Sequential(
            nn.Linear(self.enhanced_state_dim + num_districts * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(self.enhanced_state_dim + num_districts * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize optimizers and target networks
        self.actor_target = self._create_target_network(self.actor)
        self.critic1_target = self._create_target_network(self.critic1)
        self.critic2_target = self._create_target_network(self.critic2)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        
    def _create_target_network(self, network):
        """Create a target network"""
        target = type(network)(network[0].in_features, network[-2].out_features)
        target.load_state_dict(network.state_dict())
        return target
        
    def get_observation(self, raw_obs: np.ndarray) -> np.ndarray:
        """Enhance observation with global ontology knowledge"""
        
        kg_obs = self.kg.get_enhanced_observation('redistribution', 0)
        
        # Extract features for each district
        kg_features = []
        for d_id in range(self.num_districts):
            d_state = kg_obs['district_states'].get(d_id, {})
            kg_features.extend([
                1.0 if d_state.get('under_stress', False) else 0.0,
                1.0 if d_state.get('outage', False) else 0.0,
                1.0 if d_state.get('surge', False) else 0.0
            ])
            
        enhanced_obs = np.concatenate([raw_obs, np.array(kg_features)])
        
        return enhanced_obs
        
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Select redistribution action with ontology constraints"""
        
        enhanced_state = self.get_observation(state)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(enhanced_state).unsqueeze(0)
            action_probs = self.actor(state_tensor).squeeze(0).numpy()
            
        # Reshape to (num_districts, 4) for each SoC cluster
        action_probs = action_probs.reshape(self.num_districts, 4)
        
        # Get stressed districts from ontology
        stressed_mask = self.kg.get_action_mask('redistribution', 0)
        
        # Apply mask - reduce allocation to stressed districts
        for d_id in range(self.num_districts):
            if not stressed_mask[d_id]:
                action_probs[d_id, :] *= 0.1  # Reduce but don't eliminate
                
        # Renormalize each SoC cluster distribution
        for soc_idx in range(4):
            col_sum = action_probs[:, soc_idx].sum()
            if col_sum > 0:
                action_probs[:, soc_idx] /= col_sum
                
        # Add exploration noise
        if explore:
            noise = np.random.normal(0, 0.1, size=action_probs.shape)
            action_probs = action_probs + noise
            action_probs = np.clip(action_probs, 0, 1)
            
            # Renormalize
            for soc_idx in range(4):
                action_probs[:, soc_idx] /= action_probs[:, soc_idx].sum()
                
        return action_probs.flatten()


class EVCARKGReplayBuffer:
    """Prioritized replay buffer with ontology-based prioritization"""
    
    def __init__(self, capacity: int, kg_integration):
        self.capacity = capacity
        self.kg = kg_integration
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done, info=None):
        """Add experience with priority based on ontology"""
        
        # Calculate priority based on state criticality
        priority = self._calculate_priority(state, info)
        
        self.buffer.append((state, action, reward, next_state, done, info))
        self.priorities.append(priority)
        
    def _calculate_priority(self, state, info):
        """Calculate priority using ontology knowledge"""
        priority = 1.0
        
        if info:
            # Higher priority for experiences involving stressed districts
            if info.get('district_stressed', False):
                priority *= 2.0
                
            # Higher priority for critical SoC interactions
            if info.get('critical_soc_ratio', 0) > 0.1:
                priority *= 1.5
                
            # Higher priority during outage periods
            if info.get('during_outage', False):
                priority *= 1.8
                
        return priority
        
    def sample(self, batch_size: int) -> Dict:
        """Sample batch with prioritized replay"""
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Extract batch
        batch = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        for idx in indices:
            state, action, reward, next_state, done, _ = self.buffer[idx]
            batch['states'].append(state)
            batch['actions'].append(action)
            batch['rewards'].append(reward)
            batch['next_states'].append(next_state)
            batch['dones'].append(done)
            
        # Convert to numpy arrays
        for key in batch:
            batch[key] = np.array(batch[key])
            
        return batch


class EVCARKGIntegration:
    """Integration class for EVCAR-KG with the simulation"""
    
    def __init__(self):
        from evcar_kg_ontology import EVCARKGOntology
        self.kg = EVCARKGOntology()
        self.districts = {}
        self.stations = {}
        self.substations = {}
        self.soc_clusters = {}
        self.district_agents = {}
        self.redis_agent = None
        
    def initialize_from_evcar_env(self, env_config: Dict):
        """Initialize the knowledge graph from EVCAR environment configuration"""
        
        # Create districts based on EVCAR paper configuration
        district_data = [
            {'id': 0, 'type': 'Residential', 'neighbors': [1, 3, 5], 
             'coordinates': (5, 15), 'max_load': 20},
            {'id': 1, 'type': 'Industrial', 'neighbors': [0, 2, 4], 
             'coordinates': (20, 25), 'max_load': 20},
            {'id': 2, 'type': 'Industrial', 'neighbors': [1, 3], 
             'coordinates': (5, 5), 'max_load': 85},
            {'id': 3, 'type': 'Residential', 'neighbors': [0, 2, 4], 
             'coordinates': (15, 5), 'max_load': 40, 'outage': True},
            {'id': 4, 'type': 'Commercial', 'neighbors': [1, 3, 5], 
             'coordinates': (10, 0), 'max_load': 20},
            {'id': 5, 'type': 'Residential', 'neighbors': [0, 4], 
             'coordinates': (10, -10), 'max_load': 20}
        ]
        
        self.districts = self.kg.create_districts(district_data)
        
        # Create aggregate station statistics (instead of individual stations)
        station_stats = {
            0: {'total_stations': 100, 'total_queue': 0, 'avg_wait_time': 0.0, 
                'avg_utilization': 0.0, 'status': 'Operational'},
            1: {'total_stations': 130, 'total_queue': 0, 'avg_wait_time': 0.0, 
                'avg_utilization': 0.0, 'status': 'Operational'},
            2: {'total_stations': 110, 'total_queue': 0, 'avg_wait_time': 0.0, 
                'avg_utilization': 0.0, 'status': 'Operational'},
            3: {'total_stations': 190, 'total_queue': 0, 'avg_wait_time': 0.0, 
                'avg_utilization': 0.0, 'status': 'Offline'},
            4: {'total_stations': 125, 'total_queue': 0, 'avg_wait_time': 0.0, 
                'avg_utilization': 0.0, 'status': 'Operational'},
            5: {'total_stations': 90, 'total_queue': 0, 'avg_wait_time': 0.0, 
                'avg_utilization': 0.0, 'status': 'Operational'}
        }
        
        self.stations = self.kg.create_charging_stations_aggregate(station_stats)
        
        # Create SoC clusters
        self.soc_clusters = self.kg.create_soc_clusters()
        
        # Create substations
        substation_data = []
        for i in range(6):
            substation = self.kg.Substation(f"Substation_{i}")
            substation.maxCapacity = [district_data[i]['max_load']]
            substation.currentLoad = [0.0]
            substation.nodalVoltageDeviation = [0.0]
            self.substations[i] = substation
            
    def update_state(self, timestep: int, state_data: Dict):
        """Update the knowledge graph with current simulation state"""
        self.kg.update_from_simulation_state(state_data)
        
    def get_enhanced_observation(self, agent_type: str, agent_id: int) -> Dict:
        """Get enhanced observation for an agent using the knowledge graph"""
        
        if agent_type == 'district':
            return self.kg.get_district_observation(agent_id)
        elif agent_type == 'redistribution':
            # Get global state for redistribution agent
            obs = {
                'stressed_districts': [d.name for d in self.kg.query_stressed_districts()],
                'district_states': {}
            }
            
            for district_id in range(6):
                obs['district_states'][district_id] = self.kg.get_district_observation(district_id)
                
            return obs
            
        return {}
    
    def get_action_mask(self, agent_type: str, agent_id: int) -> List[bool]:
        """Get action mask based on ontology rules"""
        
        if agent_type == 'district':
            safe_actions = self.kg.get_safe_actions(agent_id)
            # Convert to boolean mask based on action space
            action_space = ['accept_critical', 'accept_low', 'accept_medium', 
                          'accept_high', 'partial_charge_50', 'partial_charge_80', 
                          'full_charge', 'reject']
            mask = [action in safe_actions for action in action_space]
            return mask
            
        elif agent_type == 'redistribution':
            # For redistribution agent, mask out stressed districts
            stressed = self.kg.query_stressed_districts()
            stressed_ids = [int(d.name.split('_')[1]) for d in stressed]
            mask = [i not in stressed_ids for i in range(6)]
            return mask
            
        return []
    
    def calculate_priority_reward(self, charging_events: List[Dict]) -> float:
        """Calculate priority-based reward using ontology"""
        
        reward = 0.0
        for event in charging_events:
            # Use aggregate SoC information instead of individual EVs
            soc_category = event.get('soc_category', 'medium')
            charge_amount = event.get('charge_amount', 0)
            
            # Higher reward for charging critical EVs
            if soc_category == 'critical':
                reward += charge_amount * 2.0
            elif soc_category == 'low':
                reward += charge_amount * 1.5
            else:
                reward += charge_amount * 1.0
                
        return reward