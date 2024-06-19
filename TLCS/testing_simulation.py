import traci
import numpy as np
import random
import timeit
import os

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, delta, yellow_duration, num_states, num_actions):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._delta=delta
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []


    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        old_action = -1 # dummy init
        old_action_arr=np.zeros(self._num_actions)

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()
            current_state=np.append(current_state, old_action_arr)

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_queue=self._get_queue_length()
            reward=-1*current_queue

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state)
            action_arr=np.zeros(self._num_actions)
            action_arr[action]=1
            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            #dynamic time change
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_action = action
            old_action_arr=action_arr

            self._reward_episode.append(reward)

        #print("Total reward:", np.sum(self._reward_episode))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time


    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length() 
            self._queue_length_episode.append(queue_length)


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["-h11", "-v11", "h12", "v12", "-h21", "-v12", "h22", "v13", "-h12", "-v21", "h13", "v22", "-h22", "-v22", "h23", "v23"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time


    def _choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        return np.argmax(self._Model.predict_one(state))


    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("1", yellow_phase_code)
        traci.trafficlight.setPhase("2", yellow_phase_code)
        traci.trafficlight.setPhase("5", yellow_phase_code)
        traci.trafficlight.setPhase("6", yellow_phase_code)


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """


        if action_number == 0:
            traci.trafficlight.setPhase("1", PHASE_NS_GREEN)
            traci.trafficlight.setPhase("2", PHASE_NS_GREEN)
            traci.trafficlight.setPhase("5", PHASE_NS_GREEN)
            traci.trafficlight.setPhase("6", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("1", PHASE_NSL_GREEN)
            traci.trafficlight.setPhase("2", PHASE_NSL_GREEN)
            traci.trafficlight.setPhase("5", PHASE_NSL_GREEN)
            traci.trafficlight.setPhase("6", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("1", PHASE_EW_GREEN)
            traci.trafficlight.setPhase("2", PHASE_EW_GREEN)
            traci.trafficlight.setPhase("5", PHASE_EW_GREEN)
            traci.trafficlight.setPhase("6", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("1", PHASE_EWL_GREEN)
            traci.trafficlight.setPhase("2", PHASE_EWL_GREEN)
            traci.trafficlight.setPhase("5", PHASE_EWL_GREEN)
            traci.trafficlight.setPhase("6", PHASE_EWL_GREEN)
    
    def _traffic(self, action_number):
        if action_number==0:
            halt_1 = traci.lane.getLastStepHaltingNumber("-v11_0")
            halt_2 = traci.lane.getLastStepHaltingNumber("-v12_0")
            halt_3 = traci.lane.getLastStepHaltingNumber("v13_0")
            halt_4 = traci.lane.getLastStepHaltingNumber("v12_0")
            halt_5 = traci.lane.getLastStepHaltingNumber("-v21_0")
            halt_6 = traci.lane.getLastStepHaltingNumber("-v22_0")
            halt_7 = traci.lane.getLastStepHaltingNumber("v23_0")
            halt_8 = traci.lane.getLastStepHaltingNumber("v22_0")
        elif action_number==1:
            halt_1 = traci.lane.getLastStepHaltingNumber("-v11_1")
            halt_2 = traci.lane.getLastStepHaltingNumber("-v12_1")
            halt_3 = traci.lane.getLastStepHaltingNumber("v13_1")
            halt_4 = traci.lane.getLastStepHaltingNumber("v12_1")
            halt_5 = traci.lane.getLastStepHaltingNumber("-v21_1")
            halt_6 = traci.lane.getLastStepHaltingNumber("-v22_1")
            halt_7 = traci.lane.getLastStepHaltingNumber("v23_1")
            halt_8 = traci.lane.getLastStepHaltingNumber("v22_1")
        elif action_number==2:
            halt_1 = traci.lane.getLastStepHaltingNumber("-h11_0")
            halt_2 = traci.lane.getLastStepHaltingNumber("-h12_0")
            halt_3 = traci.lane.getLastStepHaltingNumber("h13_0")
            halt_4 = traci.lane.getLastStepHaltingNumber("h12_0")
            halt_5 = traci.lane.getLastStepHaltingNumber("-h21_0")
            halt_6 = traci.lane.getLastStepHaltingNumber("-h22_0")
            halt_7 = traci.lane.getLastStepHaltingNumber("h23_0")
            halt_8 = traci.lane.getLastStepHaltingNumber("h22_0")
        elif action_number==3:
            halt_1 = traci.lane.getLastStepHaltingNumber("-h11_1")
            halt_2 = traci.lane.getLastStepHaltingNumber("-h12_1")
            halt_3 = traci.lane.getLastStepHaltingNumber("h13_1")
            halt_4 = traci.lane.getLastStepHaltingNumber("h12_1")
            halt_5 = traci.lane.getLastStepHaltingNumber("-h21_1")
            halt_6 = traci.lane.getLastStepHaltingNumber("-h22_1")
            halt_7 = traci.lane.getLastStepHaltingNumber("h23_1")
            halt_8 = traci.lane.getLastStepHaltingNumber("h22_1")
        length=(halt_1+halt_2+halt_3+halt_4+halt_5+halt_6+halt_7+halt_8)//8
        return length


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_1 = traci.edge.getLastStepHaltingNumber("-h11")
        halt_2 = traci.edge.getLastStepHaltingNumber("-v11")
        halt_3 = traci.edge.getLastStepHaltingNumber("h12")
        halt_4 = traci.edge.getLastStepHaltingNumber("v12")
        halt_5 = traci.edge.getLastStepHaltingNumber("-h21")
        halt_6 = traci.edge.getLastStepHaltingNumber("-v12")
        halt_7 = traci.edge.getLastStepHaltingNumber("h22")
        halt_8 = traci.edge.getLastStepHaltingNumber("v13")
        halt_9 = traci.edge.getLastStepHaltingNumber("-h12")
        halt_10 = traci.edge.getLastStepHaltingNumber("-v21")
        halt_11 = traci.edge.getLastStepHaltingNumber("h13")
        halt_12 = traci.edge.getLastStepHaltingNumber("v22")
        halt_13 = traci.edge.getLastStepHaltingNumber("-h22")
        halt_14 = traci.edge.getLastStepHaltingNumber("-v22")
        halt_15 = traci.edge.getLastStepHaltingNumber("h23")
        halt_16 = traci.edge.getLastStepHaltingNumber("v23")
        
        queue_length = halt_1 + halt_2 + halt_3 + halt_4 + halt_5 + halt_6 + halt_7 + halt_8 + halt_9 + halt_10 + halt_11 + halt_12 + halt_13 + halt_14+ halt_15 + halt_16
        return queue_length

    def _get_state(self):
        state=np.zeros(self._num_states)
        state[0] = traci.edge.getLastStepHaltingNumber("-h11")
        state[1] = traci.edge.getLastStepHaltingNumber("-v11")
        state[2] = traci.edge.getLastStepHaltingNumber("h12")
        state[3] = traci.edge.getLastStepHaltingNumber("v12")
        state[4] = traci.edge.getLastStepHaltingNumber("-h21")
        state[5] = traci.edge.getLastStepHaltingNumber("-v12")
        state[6] = traci.edge.getLastStepHaltingNumber("h22")
        state[7] = traci.edge.getLastStepHaltingNumber("v13")
        state[8] = traci.edge.getLastStepHaltingNumber("-h12")
        state[9] = traci.edge.getLastStepHaltingNumber("-v21")
        state[10] = traci.edge.getLastStepHaltingNumber("h13")
        state[11] = traci.edge.getLastStepHaltingNumber("v22")
        state[12] = traci.edge.getLastStepHaltingNumber("-h22")
        state[13] = traci.edge.getLastStepHaltingNumber("-v22")
        state[14] = traci.edge.getLastStepHaltingNumber("h23")
        state[15] = traci.edge.getLastStepHaltingNumber("v23")
        return state


    @property
    def queue_length_episode(self):
        return self._queue_length_episode


    @property
    def reward_episode(self):
        return self._reward_episode



