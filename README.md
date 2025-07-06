# EVCAR-KG
Ontology-driven Knowledge Graph for Electric Vehicle Charging Network Recovery - A Multi-Agent Reinforcement Learning Framework


# EVCAR-KG: Knowledge-Infused Multi-Agent Reinforcement Learning for EV Charging Network Recovery

This repository contains the implementation of EVCAR-KG, an ontology-driven knowledge graph enhancement for the EVCAR (Electric Vehicle Charging Quality of Service Recovery) system.

## Overview

EVCAR-KG enhances multi-agent reinforcement learning (MARL) agents with semantic knowledge to improve post-outage recovery in electric vehicle charging networks. The system integrates:
- Domain-specific ontology capturing EV charging network semantics
- Knowledge-enhanced TD3-MADDPG agents
- Ontology-based action masking and reward shaping

## Requirements

```
numpy>=1.21.0
torch>=1.10.0
owlready2>=0.37
```

## Installation

```bash
pip install numpy torch owlready2
```

## Repository Structure

```
evcar-kg/
├── evcar_kg_ontology.py          # Core ontology implementation
├── evcar_kg_agent_integration.py # MARL agent enhancement
├── evcar_kg_ontology.owl         # Saved ontology (generated)
└── README.md                      # This file
```

## Usage

### 1. Initialize the Ontology

```python
from evcar_kg_ontology import EVCARKGOntology
from evcar_kg_agent_integration import EVCARKGIntegration

# Create integration instance
kg_integration = EVCARKGIntegration()

# Initialize with EVCAR environment configuration
env_config = {
    'num_districts': 6,
    'outage_district': 3,
    'outage_duration': 62,  # 31 hours in 30-min timesteps
    'num_evs': 35000
}
kg_integration.initialize_from_evcar_env(env_config)
```

### 2. Update Ontology with Simulation State

```python
# At each timestep, update with aggregated state data
state_data = {
    'districts': {
        0: {
            'load': 15.5,           # MW
            'queue_length': 10,     # Total EVs waiting
            'nvd': 0.02,           # Nodal voltage deviation
            'ev_count': 5832,      # Total EVs in district
            'critical_count': 58,   # EVs with SoC < 5%
            'avg_soc': 45.2        # Average SoC percentage
        },
        # ... other districts
    }
}
kg_integration.update_state(timestep=10, state_data=state_data)
```

### 3. Create Knowledge-Enhanced Agents

```python
from evcar_kg_agent_integration import KnowledgeEnhancedDistrictAgent

# Create district agent with ontology enhancement
district_agent = KnowledgeEnhancedDistrictAgent(
    district_id=0,
    kg_integration=kg_integration,
    state_dim=15,      # Original EVCAR state dimension
    action_dim=4,      # 4 actions (one per SoC cluster)
    hidden_dim=256,
    lr=1e-4
)

# Get enhanced observation
raw_state = np.random.rand(15)  # From EVCAR environment
enhanced_obs = district_agent.get_observation(raw_state)

# Select action with ontology constraints
action = district_agent.select_action(raw_state, explore=True)
```

### 4. Query Ontology for Decision Support

```python
# Get enhanced observations
district_obs = kg_integration.get_enhanced_observation('district', district_id=0)
global_obs = kg_integration.get_enhanced_observation('redistribution', agent_id=0)

# Get action masks based on ontology rules
action_mask = kg_integration.get_action_mask('district', district_id=0)

# Query stressed districts
stressed_districts = kg_integration.kg.query_stressed_districts()
```

## SoC Clusters

The ontology defines four State-of-Charge (SoC) clusters for EV categorization:
- **Critical**: 0-5% (highest priority)
- **Low**: 6-20% 
- **Medium**: 21-60%
- **High**: 61-100% (lowest priority)

## Key Features

### Ontology Components
- **Classes**: District, ChargingStation, ElectricVehicle, SoCCluster, Substation, EVCAR_Agent
- **Properties**: locatedIn, hasNeighbor, powers, isInSoCCluster, managesDistrict
- **Inference Rules**: Grid stress detection, surge identification, priority vehicle flagging

### Agent Enhancements
- **State Augmentation**: Adds 10 semantic features (district type, stress status, neighbor outages, etc.)
- **Action Masking**: Prevents actions that would worsen stressed districts
- **Priority Rewards**: Weights rewards based on SoC criticality

### Scalability
- Uses aggregate representations instead of individual instances
- Stores district-level EV statistics rather than 35,000 individual EVs
- Maintains computational efficiency for real-time decision making

## Integration with EVCAR

EVCAR-KG is designed to seamlessly integrate with the existing EVCAR simulation:

1. Initialize ontology at simulation start
2. Update ontology state at each timestep
3. Agents query ontology for enhanced observations
4. Action selection incorporates ontology constraints
5. Rewards include priority-based components

## Saving and Loading

```python
# Save ontology to OWL file
kg_integration.kg.save_ontology("evcar_kg_ontology.owl")

# Load existing ontology
kg_integration.kg.load_ontology("evcar_kg_ontology.owl")
```

## Citation

If you use EVCAR-KG in your research, please cite:
```
@inproceedings{abdelfattah2025evcarkg,
  title={EVCAR-KG: A Knowledge-Infused Multi-Agent Reinforcement Learning Framework for Resilient Electric Vehicle Charging Network Recovery},
  author={Abdelfattah, Mohamed},
  booktitle={36th Forum Bauinformatik},
  year={2025},
  location={Aachen, Germany}
}
```

## License

This project is licensed under the MIT License.
