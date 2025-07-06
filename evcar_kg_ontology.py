"""
EVCAR-KG: Ontology-driven Knowledge Graph for EV Charging Network Recovery
This implementation creates the ontology described in the EVCAR-KG paper
and shows how to integrate it with the MARL agents.
"""

import numpy as np
from owlready2 import *
import datetime
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EVCARKGOntology:
    """Main class for the EVCAR-KG Ontology"""
    
    def __init__(self, ontology_iri: str = "http://www.evcar-kg.org/ontology#"):
        """Initialize the ontology"""
        self.onto = get_ontology(ontology_iri)
        
        with self.onto:
            # Define classes
            self._define_classes()
            # Define object properties
            self._define_object_properties()
            # Define data properties
            self._define_data_properties()
            # Define SWRL rules
            self._define_swrl_rules()
            
    def _define_classes(self):
        """Define all ontology classes"""
        
        # Infrastructure Components
        class InfrastructureComponent(Thing):
            pass
        
        class GridEntity(Thing):
            pass
        
        class District(GridEntity):
            pass
        
        class ChargingStation(InfrastructureComponent):
            pass
        
        class ElectricVehicle(InfrastructureComponent):
            pass
        
        # SoC Clusters
        class SoCCluster(Thing):
            pass
        
        # Power Grid Components
        class PowerGridComponent(Thing):
            pass
        
        class Substation(PowerGridComponent):
            pass
        
        class TransmissionLine(PowerGridComponent):
            pass
        
        # Events and States
        class PowerOutage(Thing):
            pass
        
        class QoSRecoveryState(Thing):
            pass
        
        class ChargingSession(Thing):
            pass
        
        # Agent-related classes
        class EVCAR_Agent(Thing):
            pass
        
        class DistrictAgent(EVCAR_Agent):
            pass
        
        class RedistributionAgent(EVCAR_Agent):
            pass
        
        class Observation(Thing):
            pass
        
        class Action(Thing):
            pass
        
        class Policy(Thing):
            pass
        
        class Goal(Thing):
            pass
        
        class RewardFunction(Thing):
            pass
        
        # Store references
        self.District = District
        self.ChargingStation = ChargingStation
        self.ElectricVehicle = ElectricVehicle
        self.SoCCluster = SoCCluster
        self.Substation = Substation
        self.TransmissionLine = TransmissionLine
        self.PowerOutage = PowerOutage
        self.QoSRecoveryState = QoSRecoveryState
        self.ChargingSession = ChargingSession
        self.DistrictAgent = DistrictAgent
        self.RedistributionAgent = RedistributionAgent
        
    def _define_object_properties(self):
        """Define object properties (relationships)"""
        
        # Location properties
        class locatedIn(ObjectProperty):
            domain = [self.ChargingStation]
            range = [self.District]
        
        class hasNeighbor(ObjectProperty, SymmetricProperty):
            domain = [self.District]
            range = [self.District]
        
        # Power relationships
        class powers(ObjectProperty):
            domain = [self.Substation]
            range = [self.ChargingStation]
        
        class suppliedBy(ObjectProperty):
            domain = [self.District]
            range = [self.Substation]
        
        # Agent relationships
        class managesDistrict(ObjectProperty):
            domain = [self.DistrictAgent]
            range = [self.District]
        
        class observesNetworkState(ObjectProperty):
            domain = [self.RedistributionAgent]
            range = [self.QoSRecoveryState]
        
        # EV relationships
        class isInSoCCluster(ObjectProperty):
            domain = [self.ElectricVehicle]
            range = [self.SoCCluster]
        
        # Event relationships
        class affectsDistrict(ObjectProperty):
            domain = [self.PowerOutage]
            range = [self.District]
        
        class involvesVehicle(ObjectProperty):
            domain = [self.ChargingSession]
            range = [self.ElectricVehicle]
        
        class atStation(ObjectProperty):
            domain = [self.ChargingSession]
            range = [self.ChargingStation]
        
        # Store references
        self.locatedIn = locatedIn
        self.hasNeighbor = hasNeighbor
        self.powers = powers
        self.suppliedBy = suppliedBy
        self.managesDistrict = managesDistrict
        self.observesNetworkState = observesNetworkState
        self.isInSoCCluster = isInSoCCluster
        self.affectsDistrict = affectsDistrict
        
    def _define_data_properties(self):
        """Define data properties (attributes)"""
        
        # District properties
        class districtType(DataProperty):
            domain = [self.District]
            range = [str]
        
        class currentLoad(DataProperty):
            domain = [self.District]
            range = [float]
        
        class queueLength(DataProperty):
            domain = [self.District]
            range = [int]
        
        class congestionLevel(DataProperty):
            domain = [self.District]
            range = [str]  # LOW, MEDIUM, HIGH
        
        class outageStatus(DataProperty):
            domain = [self.District]
            range = [bool]
        
        class surgeStatus(DataProperty):
            domain = [self.District]
            range = [bool]
        
        # Charging Station properties
        class numberOfChargers(DataProperty):
            domain = [self.ChargingStation]
            range = [int]
        
        class capacity(DataProperty):
            domain = [self.ChargingStation]
            range = [float]
        
        class currentQueueLength(DataProperty):
            domain = [self.ChargingStation]
            range = [int]
        
        class averageWaitingTime(DataProperty):
            domain = [self.ChargingStation]
            range = [float]
        
        class utilizationFactor(DataProperty):
            domain = [self.ChargingStation]
            range = [float]
        
        class currentPowerLoad(DataProperty):
            domain = [self.ChargingStation]
            range = [float]
        
        class operationalStatus(DataProperty):
            domain = [self.ChargingStation]
            range = [str]  # Operational, Offline
        
        # EV properties
        class hasBatteryCapacity(DataProperty):
            domain = [self.ElectricVehicle]
            range = [float]
        
        class hasStateOfCharge(DataProperty):
            domain = [self.ElectricVehicle]
            range = [float]
        
        # SoC Cluster properties
        class lowerBound(DataProperty):
            domain = [self.SoCCluster]
            range = [float]
        
        class upperBound(DataProperty):
            domain = [self.SoCCluster]
            range = [float]
        
        # Substation properties
        class maxCapacity(DataProperty):
            domain = [self.Substation]
            range = [float]
        
        class nodalVoltageDeviation(DataProperty):
            domain = [self.Substation]
            range = [float]
        
        # Store references
        self.districtType = districtType
        self.currentLoad = currentLoad
        self.queueLength = queueLength
        self.congestionLevel = congestionLevel
        self.outageStatus = outageStatus
        self.surgeStatus = surgeStatus
        self.numberOfChargers = numberOfChargers
        self.currentQueueLength = currentQueueLength
        self.averageWaitingTime = averageWaitingTime
        self.utilizationFactor = utilizationFactor
        self.operationalStatus = operationalStatus
        self.hasBatteryCapacity = hasBatteryCapacity
        self.hasStateOfCharge = hasStateOfCharge
        self.maxCapacity = maxCapacity
        self.nodalVoltageDeviation = nodalVoltageDeviation
        
    def _define_swrl_rules(self):
        """Define SWRL rules for inference"""
        
        # Rule 1: Grid Stress Inference
        # If a District has voltage deviation > 0.04 p.u. and more than 20 EVs in queue
        # then mark it as isUnderStress = true
        rule1 = Imp()
        rule1.set_as_rule("""
            District(?d) ^ queueLength(?d, ?q) ^ nodalVoltageDeviation(?d, ?v) ^
            swrlb:greaterThan(?v, 0.04) ^ swrlb:greaterThan(?q, 20)
            -> isUnderStress(?d, true)
        """)
        
        # Rule 2: Surge Detection
        # If a District has a neighbor with outageStatus = true and its own
        # queueLength increased by >50% within one hour then set surgeStatus = true
        rule2 = Imp()
        rule2.set_as_rule("""
            District(?d1) ^ District(?d2) ^ hasNeighbor(?d1, ?d2) ^
            outageStatus(?d2, true) ^ queueLength(?d1, ?q) ^
            previousQueueLength(?d1, ?pq) ^
            swrlb:greaterThan(?q, ?pq * 1.5)
            -> surgeStatus(?d1, true)
        """)
        
        # Rule 3: Priority Vehicle Identification
        # If an ElectricVehicle's SoC is in CriticalSoC then tag as highPriority
        rule3 = Imp()
        rule3.set_as_rule("""
            ElectricVehicle(?ev) ^ hasStateOfCharge(?ev, ?soc) ^
            swrlb:lessThan(?soc, 5.0)
            -> highPriority(?ev, true)
        """)
        
    def create_districts(self, district_data: List[Dict]) -> Dict[int, object]:
        """Create district individuals from simulation data"""
        districts = {}
        
        for d in district_data:
            district = self.District(f"District_{d['id']}")
            district.districtType = [d['type']]  # Residential, Commercial, Industrial
            district.currentLoad = [d.get('load', 0.0)]
            district.queueLength = [d.get('queue_length', 0)]
            district.congestionLevel = [d.get('congestion', 'LOW')]
            district.outageStatus = [d.get('outage', False)]
            district.surgeStatus = [d.get('surge', False)]
            districts[d['id']] = district
            
        # Set neighbor relationships
        for d in district_data:
            for neighbor_id in d.get('neighbors', []):
                districts[d['id']].hasNeighbor.append(districts[neighbor_id])
                
        return districts
    
    def create_charging_stations(self, station_data: List[Dict], districts: Dict) -> Dict[int, object]:
        """Create charging station individuals"""
        stations = {}
        
        for s in station_data:
            station = self.ChargingStation(f"Station_{s['id']}")
            station.numberOfChargers = [s['num_chargers']]
            station.capacity = [s.get('capacity', 36.0)]  # 36 kW default
            station.currentQueueLength = [s.get('queue', 0)]
            station.averageWaitingTime = [s.get('wait_time', 0.0)]
            station.utilizationFactor = [s.get('utilization', 0.0)]
            station.currentPowerLoad = [s.get('power_load', 0.0)]
            station.operationalStatus = [s.get('status', 'Operational')]
            
            # Link to district
            if s['district_id'] in districts:
                station.locatedIn = [districts[s['district_id']]]
                
            stations[s['id']] = station
            
        return stations
    
    def create_soc_clusters(self) -> Dict[str, object]:
        """Create SoC cluster individuals as defined in the paper"""
        clusters = {}
        
        # Critical SoC: 0-10%
        critical = self.SoCCluster("CriticalSoC")
        critical.lowerBound = [0.0]
        critical.upperBound = [10.0]
        clusters['critical'] = critical
        
        # Low SoC: 20-50%
        low = self.SoCCluster("LowSoC")
        low.lowerBound = [20.0]
        low.upperBound = [50.0]
        clusters['low'] = low
        
        # Medium SoC: 50-80%
        medium = self.SoCCluster("MediumSoC")
        medium.lowerBound = [50.0]
        medium.upperBound = [80.0]
        clusters['medium'] = medium
        
        # High SoC: >80%
        high = self.SoCCluster("HighSoC")
        high.lowerBound = [80.0]
        high.upperBound = [100.0]
        clusters['high'] = high
        
        return clusters
    
    def create_substations(self, substation_data: List[Dict], stations: Dict) -> Dict[int, object]:
        """Create substation individuals"""
        substations = {}
        
        for sub in substation_data:
            substation = self.Substation(f"Substation_{sub['id']}")
            substation.maxCapacity = [sub['max_capacity']]
            substation.currentLoad = [sub.get('current_load', 0.0)]
            substation.nodalVoltageDeviation = [sub.get('nvd', 0.0)]
            
            # Link to stations it powers
            for station_id in sub.get('powers_stations', []):
                if station_id in stations:
                    substation.powers.append(stations[station_id])
                    
            substations[sub['id']] = substation
            
        return substations
    
    def create_agents(self, districts: Dict) -> Tuple[Dict, object]:
        """Create agent individuals"""
        district_agents = {}
        
        # Create district agents
        for district_id, district in districts.items():
            agent = self.DistrictAgent(f"DistrictAgent_{district_id}")
            agent.managesDistrict = [district]
            district_agents[district_id] = agent
            
        # Create redistribution agent
        redis_agent = self.RedistributionAgent("RedistributionAgent_Central")
        
        # Create recovery state
        recovery_state = self.QoSRecoveryState("CurrentRecoveryState")
        redis_agent.observesNetworkState = [recovery_state]
        
        return district_agents, redis_agent
    
    def query_stressed_districts(self) -> List:
        """Query for districts under stress"""
        # This would run the SWRL reasoner and return stressed districts
        # For now, we'll use a simple query
        stressed = []
        for district in self.onto.District.instances():
            if (hasattr(district, 'nodalVoltageDeviation') and 
                hasattr(district, 'queueLength')):
                if (district.nodalVoltageDeviation[0] > 0.04 and 
                    district.queueLength[0] > 20):
                    stressed.append(district)
        return stressed
    
    def query_priority_evs(self) -> List:
        """Query for priority EVs (critical SoC)"""
        priority_evs = []
        for ev in self.onto.ElectricVehicle.instances():
            if hasattr(ev, 'hasStateOfCharge'):
                if ev.hasStateOfCharge[0] < 5.0:  # Critical threshold
                    priority_evs.append(ev)
        return priority_evs
    
    def update_district_state(self, district_id: int, state_data: Dict):
        """Update district state in the ontology"""
        district = self.onto.search_one(iri=f"*District_{district_id}")
        if district:
            if 'load' in state_data:
                district.currentLoad = [state_data['load']]
            if 'queue_length' in state_data:
                district.queueLength = [state_data['queue_length']]
            if 'congestion' in state_data:
                district.congestionLevel = [state_data['congestion']]
            if 'nvd' in state_data:
                district.nodalVoltageDeviation = [state_data['nvd']]
                
    def get_district_observation(self, district_id: int) -> Dict:
        """Get enhanced observation for a district agent"""
        district = self.onto.search_one(iri=f"*District_{district_id}")
        if not district:
            return {}
            
        # Basic observation
        obs = {
            'type': district.districtType[0] if hasattr(district, 'districtType') else 'Unknown',
            'load': district.currentLoad[0] if hasattr(district, 'currentLoad') else 0.0,
            'queue': district.queueLength[0] if hasattr(district, 'queueLength') else 0,
            'congestion': district.congestionLevel[0] if hasattr(district, 'congestionLevel') else 'LOW',
            'outage': district.outageStatus[0] if hasattr(district, 'outageStatus') else False,
            'surge': district.surgeStatus[0] if hasattr(district, 'surgeStatus') else False,
        }
        
        # Add neighbor information
        neighbor_info = []
        if hasattr(district, 'hasNeighbor'):
            for neighbor in district.hasNeighbor:
                neighbor_info.append({
                    'id': neighbor.name.split('_')[1],
                    'outage': neighbor.outageStatus[0] if hasattr(neighbor, 'outageStatus') else False,
                    'congestion': neighbor.congestionLevel[0] if hasattr(neighbor, 'congestionLevel') else 'LOW'
                })
        obs['neighbors'] = neighbor_info
        
        # Check if under stress (would be inferred by reasoner)
        obs['under_stress'] = self._check_stress(district)
        
        return obs
    
    def _check_stress(self, district) -> bool:
        """Check if district is under stress based on rules"""
        if hasattr(district, 'nodalVoltageDeviation') and hasattr(district, 'queueLength'):
            return (district.nodalVoltageDeviation[0] > 0.04 and 
                    district.queueLength[0] > 20)
        return False
    
    def get_safe_actions(self, district_id: int) -> List[str]:
        """Get safe actions for a district based on ontology rules"""
        district = self.onto.search_one(iri=f"*District_{district_id}")
        if not district:
            return []
            
        safe_actions = []
        
        # If district is under stress, avoid accepting more EVs
        if not self._check_stress(district):
            safe_actions.extend(['accept_critical', 'accept_low', 'accept_medium'])
        else:
            # Only accept critical EVs when stressed
            safe_actions.append('accept_critical')
            
        # Always allow partial charging strategies
        safe_actions.extend(['partial_charge_50', 'partial_charge_80'])
        
        return safe_actions
    
    def save_ontology(self, filepath: str):
        """Save the ontology to file"""
        self.onto.save(file=filepath, format="rdfxml")
        logger.info(f"Ontology saved to {filepath}")
        
    def load_ontology(self, filepath: str):
        """Load ontology from file"""
        self.onto = get_ontology(filepath).load()
        logger.info(f"Ontology loaded from {filepath}")


# Example usage integrated with EVCAR simulation
class EVCARKGIntegration:
    """Integration class for EVCAR-KG with the simulation"""
    
    def __init__(self):
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
             'coordinates': (15, 5), 'max_load': 40, 'outage': True},  # Outage district
            {'id': 4, 'type': 'Commercial', 'neighbors': [1, 3, 5], 
             'coordinates': (10, 0), 'max_load': 20},
            {'id': 5, 'type': 'Residential', 'neighbors': [0, 4], 
             'coordinates': (10, -10), 'max_load': 20}
        ]
        
        self.districts = self.kg.create_districts(district_data)
        
        # Create charging stations
        station_data = []
        station_counts = [100, 130, 110, 190, 125, 90]  # From paper
        station_id = 0
        for district_id, count in enumerate(station_counts):
            for i in range(count):
                station_data.append({
                    'id': station_id,
                    'district_id': district_id,
                    'num_chargers': 1,  # Simplified, could be more
                    'capacity': 36.0,  # 36 kW fast chargers
                    'status': 'Offline' if district_id == 3 else 'Operational'
                })
                station_id += 1
                
        self.stations = self.kg.create_charging_stations(station_data, self.districts)
        
        # Create SoC clusters
        self.soc_clusters = self.kg.create_soc_clusters()
        
        # Create substations (simplified)
        substation_data = []
        for i in range(6):
            substation_data.append({
                'id': i,
                'max_capacity': district_data[i]['max_load'],
                'powers_stations': [s['id'] for s in station_data if s['district_id'] == i]
            })
            
        self.substations = self.kg.create_substations(substation_data, self.stations)
        
        # Create agents
        self.district_agents, self.redis_agent = self.kg.create_agents(self.districts)
        
        logger.info("Knowledge graph initialized from EVCAR environment")
        
    def update_state(self, timestep: int, state_data: Dict):
        """Update the knowledge graph with current simulation state"""
        
        for district_id, district_state in state_data.get('districts', {}).items():
            self.kg.update_district_state(district_id, district_state)
            
        # Update station states
        for station_id, station_state in state_data.get('stations', {}).items():
            station = self.kg.onto.search_one(iri=f"*Station_{station_id}")
            if station:
                if 'queue' in station_state:
                    station.currentQueueLength = [station_state['queue']]
                if 'utilization' in station_state:
                    station.utilizationFactor = [station_state['utilization']]
                if 'wait_time' in station_state:
                    station.averageWaitingTime = [station_state['wait_time']]
                    
    def get_enhanced_observation(self, agent_type: str, agent_id: int) -> Dict:
        """Get enhanced observation for an agent using the knowledge graph"""
        
        if agent_type == 'district':
            return self.kg.get_district_observation(agent_id)
        elif agent_type == 'redistribution':
            # Get global state for redistribution agent
            obs = {
                'stressed_districts': [d.name for d in self.kg.query_stressed_districts()],
                'priority_evs_count': len(self.kg.query_priority_evs()),
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
            ev_id = event['ev_id']
            ev = self.kg.onto.search_one(iri=f"*EV_{ev_id}")
            
            if ev and hasattr(ev, 'hasStateOfCharge'):
                soc = ev.hasStateOfCharge[0]
                
                # Higher reward for charging critical EVs
                if soc < 10.0:  # Critical
                    reward += event['charge_amount'] * 2.0
                elif soc < 50.0:  # Low
                    reward += event['charge_amount'] * 1.5
                else:
                    reward += event['charge_amount'] * 1.0
                    
        return reward


# Example usage script
if __name__ == "__main__":
    # Initialize the integration
    evcar_kg = EVCARKGIntegration()
    
    # Initialize from EVCAR environment configuration
    env_config = {
        'num_districts': 6,
        'outage_district': 3,
        'outage_duration': 62,  # 31 hours in 30-min timesteps
        'num_evs': 35000
    }
    
    evcar_kg.initialize_from_evcar_env(env_config)
    
    # Example: Update state at timestep t
    state_data = {
        'districts': {
            0: {'load': 15.5, 'queue_length': 10, 'nvd': 0.02},
            1: {'load': 18.2, 'queue_length': 25, 'nvd': 0.045},  # Will be stressed
            2: {'load': 45.0, 'queue_length': 15, 'nvd': 0.03},
            3: {'load': 0.0, 'queue_length': 0, 'nvd': 0.0, 'outage': True},
            4: {'load': 12.0, 'queue_length': 8, 'nvd': 0.015},
            5: {'load': 16.0, 'queue_length': 30, 'nvd': 0.05}  # Will be stressed
        }
    }
    
    evcar_kg.update_state(timestep=10, state_data=state_data)
    
    # Get enhanced observations for agents
    district_obs = evcar_kg.get_enhanced_observation('district', 1)
    print(f"District 1 observation: {district_obs}")
    
    redis_obs = evcar_kg.get_enhanced_observation('redistribution', 0)
    print(f"Redistribution agent observation: {redis_obs}")
    
    # Get action masks
    district_mask = evcar_kg.get_action_mask('district', 1)
    print(f"District 1 action mask: {district_mask}")
    
    redis_mask = evcar_kg.get_action_mask('redistribution', 0)
    print(f"Redistribution action mask: {redis_mask}")
    
    # Save the ontology
    evcar_kg.kg.save_ontology("evcar_kg_ontology.owl")
