"""
Example integration of EVCAR-KG with simulation environment
"""

from evcar_kg_ontology import EVCARKGOntology
from evcar_kg_agent_integration import EVCARKGIntegration

# Initialize
kg_integration = EVCARKGIntegration()

# Configure environment
env_config = {
    'num_districts': 6,
    'outage_district': 3,
    'outage_duration': 62,
    'num_evs': 35000
}

# Initialize from environment
kg_integration.initialize_from_evcar_env(env_config)

# Example state update
state_data = {
    'districts': {
        0: {'load': 15.5, 'queue_length': 10, 'nvd': 0.02, 
            'ev_count': 5832, 'critical_count': 58, 'avg_soc': 45.2}
    }
}

kg_integration.update_state(timestep=10, state_data=state_data)
print("Ontology initialized and updated successfully!")