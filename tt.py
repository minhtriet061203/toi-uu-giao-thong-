import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pygame
import sys
import time
import math


class TrafficNetwork:
    def __init__(self, congestion_data):
        """
        Initialize traffic network with 6 nodes (A, B, C, D, E, F)
        congestion_data: dictionary containing congestion indices for each road at each intersection
        """
        self.congestion_data = congestion_data
        self.intersections = ["A", "B", "C", "D", "E", "F"]
        self.roads_per_intersection = ["1", "2", "3", "4"]  # Each intersection has 4 roads

        # Basic parameters
        self.min_green_time = 15  # Minimum green light time (seconds)
        self.max_green_time = 90  # Maximum green light time (seconds)
        self.amber_time = 3  # Amber light time (seconds)
        self.clearance_time = 2  # Intersection clearance time (seconds)

        # Distance matrix between intersections (units: meters)
        self.distance_matrix = {
            "A": {"A": 0, "B": 300, "C": 0, "D": 0, "E": 0, "F": 250},
            "B": {"A": 300, "B": 0, "C": 280, "D": 0, "E": 320, "F": 0},
            "C": {"A": 0, "B": 280, "C": 0, "D": 350, "E": 0, "F": 0},
            "D": {"A": 0, "B": 0, "C": 350, "D": 0, "E": 0, "F": 0},
            "E": {"A": 0, "B": 320, "C": 0, "D": 0, "E": 0, "F": 270},
            "F": {"A": 250, "B": 0, "C": 0, "D": 0, "E": 270, "F": 0}
        }

        # Connection matrix between intersections
        # Value: (output road from node1, input road to node2)
        self.connections = {
            "A": {"A": None, "B": ("2", "1"), "C": None, "D": None, "E": None, "F": ("1", "1")},
            "B": {"A": ("1", "2"), "B": None, "C": ("2", "1"), "D": None, "E": ("3", "2"), "F": None},
            "C": {"A": None, "B": ("1", "2"), "C": None, "D": ("2", "1"), "E": None, "F": None},
            "D": {"A": ("1", "3"), "B": None, "C": ("1", "2"), "D": None, "E": None, "F": None},
            "E": {"A": None, "B": ("2", "3"), "C": None, "D": None, "E": None, "F": ("1", "2")},
            "F": {"A": ("1", "1"), "B": None, "C": None, "D": None, "E": ("2", "1"), "F": None}
        }

        # Setup phase groups for each intersection
        # (roads 1,3 in one phase and 2,4 in another phase)
        self.phase_groups = {
            intersection: [
                [f"{intersection}-1", f"{intersection}-3"],
                [f"{intersection}-2", f"{intersection}-4"]
            ] for intersection in self.intersections
        }

        # Average vehicle speed (m/s) - about 30km/h
        self.avg_speed = 8.33

        # Result variables
        self.signal_timings = {}
        self.cycle_times = {}
        self.offset_times = {}

    def optimize_independent(self):
        """
        Optimize each intersection independently without considering relationships
        """
        for intersection in self.intersections:
            # Get congestion data for this intersection
            congestion_at_intersection = {}
            for road in self.roads_per_intersection:
                key = f"{intersection}-{road}"
                congestion_at_intersection[key] = self.congestion_data[key]

            # Optimize this intersection
            timings, cycle_time = self.optimize_single_intersection(congestion_at_intersection, intersection)

            # Save the results
            self.signal_timings[intersection] = timings
            self.cycle_times[intersection] = cycle_time

        return self.signal_timings, self.cycle_times

    def optimize_network(self):
        """
        Optimize the entire traffic network with 6 connected intersections
        """
        # Step 1: Optimize each intersection independently
        self.optimize_independent()

        # Step 2: Calculate offsets to create green waves
        self.calculate_offsets()

        # Step 3: Adjust cycles between intersections
        self.normalize_cycle_times()

        # Step 4: Adjust green times after common cycle
        self.adjust_green_times()

        return self.signal_timings, self.cycle_times, self.offset_times

    def optimize_single_intersection(self, congestion_levels, intersection):
        """
        Algorithm to optimize traffic light timing for 1 intersection
        using Webster's method combined with Fuzzy Logic
        """
        # Get phase groups for this intersection
        phase_groups = self.phase_groups[intersection]

        # 1. Calculate average density for each phase
        phase_congestion = []
        for phase in phase_groups:
            # Get the maximum congestion in the phase (determining factor)
            max_congestion = max([congestion_levels[road] for road in phase])
            phase_congestion.append(max_congestion)

        # 2. Apply Fuzzy Logic to adjust light times
        # Setup linguistic variables for Fuzzy Logic
        congestion = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'congestion')
        green_time = ctrl.Consequent(np.arange(self.min_green_time, self.max_green_time + 1, 1), 'green_time')

        # Define fuzzy sets for congestion density
        congestion['low'] = fuzz.trimf(congestion.universe, [0, 0, 0.3])
        congestion['medium'] = fuzz.trimf(congestion.universe, [0.2, 0.5, 0.8])
        congestion['high'] = fuzz.trimf(congestion.universe, [0.7, 1, 1])

        # Define fuzzy sets for green light time
        green_time['short'] = fuzz.trimf(green_time.universe, [self.min_green_time, self.min_green_time, 35])
        green_time['medium'] = fuzz.trimf(green_time.universe, [30, 45, 60])
        green_time['long'] = fuzz.trimf(green_time.universe, [55, self.max_green_time, self.max_green_time])

        # Define fuzzy rules
        rule1 = ctrl.Rule(congestion['low'], green_time['short'])
        rule2 = ctrl.Rule(congestion['medium'], green_time['medium'])
        rule3 = ctrl.Rule(congestion['high'], green_time['long'])

        # Setup fuzzy control system
        green_time_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
        green_time_simulator = ctrl.ControlSystemSimulation(green_time_ctrl)

        # 3. Apply Webster's formula to calculate optimal cycle
        lost_time_per_phase = self.amber_time + self.clearance_time
        total_lost_time = lost_time_per_phase * len(phase_groups)

        # Calculate congestion ratio for each phase
        y_values = [min(0.95 * congestion, 0.95) for congestion in phase_congestion]
        Y = sum(y_values)

        # Webster's formula for optimal cycle
        optimal_cycle = (1.5 * total_lost_time + 5) / (1 - Y)
        optimal_cycle = min(max(optimal_cycle, 60), 150)  # Limit cycle to 60-150s

        # 4. Allocate green time for each phase
        effective_green_times = {}
        timings = {}

        for i, phase in enumerate(phase_groups):
            # Use Fuzzy Logic to calculate initial gr   een time
            green_time_simulator.input['congestion'] = phase_congestion[i]
            green_time_simulator.compute()
            initial_green = green_time_simulator.output['green_time']

            # Adjust according to Webster's formula
            effective_green = (y_values[i] / Y) * (optimal_cycle - total_lost_time)

            # Take weighted average
            final_green = 0.7 * effective_green + 0.3 * initial_green

            # Ensure green time is within limits
            final_green = min(max(final_green, self.min_green_time), self.max_green_time)
            effective_green_times[i] = round(final_green)   

        # 5. Calculate red time for each phase
        for i, phase in enumerate(phase_groups):
            for road in phase:
                other_phases_green = sum([effective_green_times[j] for j in range(len(phase_groups)) if j != i])
                red_time = other_phases_green + lost_time_per_phase * (len(phase_groups) - 1)

                timings[road] = {
                    "green_time": effective_green_times[i],
                    "red_time": round(red_time),
                    "amber_time": self.amber_time,
                    "clearance_time": self.clearance_time
                }

        # 6. Calculate and return cycle information
        cycle_time = sum(
            [effective_green_times[i] + self.amber_time for i in range(len(phase_groups))]) + total_lost_time

        return timings, round(cycle_time)

    def calculate_offsets(self):
        """
        Calculate phase offsets between intersections to create green waves
        """
        # Choose the intersection with highest traffic flow as the reference (offset = 0)
        total_congestion_per_intersection = {}
        for intersection in self.intersections:
            total = sum(self.congestion_data[f"{intersection}-{road}"] for road in self.roads_per_intersection)
            total_congestion_per_intersection[intersection] = total

        reference_intersection = max(total_congestion_per_intersection, key=total_congestion_per_intersection.get)

        # Set reference intersection offset to 0
        self.offset_times = {reference_intersection: 0}

        # Calculate offsets for other intersections
        for intersection in self.intersections:
            if intersection == reference_intersection:
                continue

            # Find connection between reference and this intersection
            connection = self.connections[reference_intersection][intersection]
            if connection is None:
                # If no direct connection, find the shortest path
                path = self.find_shortest_path(reference_intersection, intersection)
                if not path:
                    # If no path, set offset to 0
                    self.offset_times[intersection] = 0
                    continue

                # Calculate total travel time
                total_travel_time = 0
                for i in range(len(path) - 1):
                    distance = self.distance_matrix[path[i]][path[i + 1]]
                    total_travel_time += distance / self.avg_speed

                # Calculate offset based on travel time
                # Goal: Vehicles arrive at the next intersection when green light starts
                cycle_time = self.cycle_times[intersection]
                offset = total_travel_time % cycle_time
                self.offset_times[intersection] = round(offset)
            else:
                # If direct connection, calculate offset based on travel time
                out_road, in_road = connection
                distance = self.distance_matrix[reference_intersection][intersection]
                travel_time = distance / self.avg_speed

                # Calculate green start time for input road at destination
                in_road_key = f"{intersection}-{in_road}"
                phase_with_in_road = None
                for phase in self.phase_groups[intersection]:
                    if in_road_key in phase:
                        phase_with_in_road = phase
                        break

                # Green light start time for in_road at destination
                if phase_with_in_road == self.phase_groups[intersection][0]:
                    # If it's phase 1
                    green_start_time_dest = 0
                else:
                    # If it's phase 2
                    green_start_time_dest = self.signal_timings[intersection][phase_with_in_road[0]][
                                                "green_time"] + self.amber_time

                # Green light start time for out_road at source
                out_road_key = f"{reference_intersection}-{out_road}"
                phase_with_out_road = None
                for phase in self.phase_groups[reference_intersection]:
                    if out_road_key in phase:
                        phase_with_out_road = phase
                        break

                if phase_with_out_road == self.phase_groups[reference_intersection][0]:
                    # If it's phase 1
                    green_start_time_source = 0
                else:
                    # If it's phase 2
                    green_start_time_source = self.signal_timings[reference_intersection][phase_with_out_road[0]][
                                                  "green_time"] + self.amber_time

                # Calculate offset: green light time at destination = green light time at source + travel time
                cycle_time = self.cycle_times[intersection]
                ideal_offset = (green_start_time_source + travel_time - green_start_time_dest) % cycle_time
                self.offset_times[intersection] = round(ideal_offset)

        return self.offset_times

    def normalize_cycle_times(self):
        """
        Adjust cycles of intersections to synchronize with each other
        """
        # Find the largest cycle among intersections
        max_cycle = max(self.cycle_times.values())

        # Set common cycle to the largest cycle
        for intersection in self.intersections:
            self.cycle_times[intersection] = max_cycle

        return self.cycle_times

    def adjust_green_times(self):
        """
        Adjust green light times after setting a common cycle
        """
        for intersection in self.intersections:
            cycle_time = self.cycle_times[intersection]
            phase_groups = self.phase_groups[intersection]

            # Calculate current total green time
            current_green_times = [self.signal_timings[intersection][phase[0]]["green_time"] for phase in phase_groups]
            total_current_green = sum(current_green_times)

            # Calculate lost time in the cycle
            lost_time_per_phase = self.amber_time + self.clearance_time
            total_lost_time = lost_time_per_phase * len(phase_groups)

            # Calculate total green time possible in the new cycle
            available_green_time = cycle_time - total_lost_time

            # Calculate adjustment ratio
            adjustment_ratio = available_green_time / total_current_green

            # Adjust green time for each phase
            for i, phase in enumerate(phase_groups):
                new_green_time = round(current_green_times[i] * adjustment_ratio)
                new_green_time = max(new_green_time, self.min_green_time)  # Ensure minimum green time

                # Update new green time
                for road in phase:
                    self.signal_timings[intersection][road]["green_time"] = new_green_time

            # Recalculate red time for each phase
            for i, phase in enumerate(phase_groups):
                for road in phase:
                    # Green time of other phases + amber time of other phases
                    other_phases_green = sum([self.signal_timings[intersection][phase_groups[j][0]]["green_time"]
                                              for j in range(len(phase_groups)) if j != i])
                    other_phases_amber = (len(phase_groups) - 1) * self.amber_time

                    # Red time = cycle - green time of this phase - amber time of this phase
                    red_time = cycle_time - self.signal_timings[intersection][road]["green_time"] - self.amber_time
                    self.signal_timings[intersection][road]["red_time"] = red_time

        return self.signal_timings

    def find_shortest_path(self, start, end):
        """
        Find the shortest path between two intersections
        """
        # Dijkstra's algorithm
        distances = {intersection: float('infinity') for intersection in self.intersections}
        distances[start] = 0

        unvisited = list(self.intersections)
        previous = {intersection: None for intersection in self.intersections}

        while unvisited:
            current = min(unvisited, key=lambda node: distances[node])

            if current == end:
                # Create path from end to start
                path = []
                while current:
                    path.append(current)
                    current = previous[current]
                # Reverse the path
                return path[::-1]

            if distances[current] == float('infinity'):
                # No path to the destination
                break

            # Remove current node from unvisited list
            unvisited.remove(current)

            # Check neighboring nodes
            for neighbor in self.intersections:
                if neighbor == current:
                    continue

                if self.connections[current][neighbor]:
                    # There is a path from current to neighbor
                    distance = self.distance_matrix[current][neighbor]
                    alternative_route = distances[current] + distance

                    if alternative_route < distances[neighbor]:
                        distances[neighbor] = alternative_route
                        previous[neighbor] = current

        # If no path found
        return []


def run_network_simulation(network):
    """
    Visual simulation of traffic light network operation
    """
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("Smart Traffic Light Network Simulation")
    clock = pygame.time.Clock()

    # Fonts
    font_small = pygame.font.Font(None, 24)
    font_medium = pygame.font.Font(None, 32)
    font_large = pygame.font.Font(None, 40)

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (150, 150, 150)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    LIGHT_BLUE = (173, 216, 230)

    # Position of intersections - adjusted for 6 nodes
    intersection_positions = {
        "A": (250, 250),
        "B": (250, 550),
        "C": (550, 550),
        "D": (850, 550),
        "E": (550, 250),
        "F": (850, 250)
    }

    # Size of each intersection
    intersection_size = 120

    # Position of traffic lights for each intersection
    traffic_light_positions = {}
    for intersection, center in intersection_positions.items():
        x, y = center
        traffic_light_positions[f"{intersection}-1"] = (x - intersection_size // 2, y)
        traffic_light_positions[f"{intersection}-2"] = (x, y - intersection_size // 2)
        traffic_light_positions[f"{intersection}-3"] = (x + intersection_size // 2, y)
        traffic_light_positions[f"{intersection}-4"] = (x, y + intersection_size // 2)

    # Get information from network
    signal_timings = network.signal_timings
    cycle_times = network.cycle_times
    offset_times = network.offset_times
    congestion_data = network.congestion_data

    # Draw intersection
    def draw_intersection(center, size, intersection_name, time_in_cycle):
        x, y = center

        # Draw square representing the intersection
        pygame.draw.rect(screen, BLACK, (x - size // 2, y - size // 2, size, size))

        # Draw vertical and horizontal roads through the intersection
        road_width = 30
        pygame.draw.rect(screen, GRAY, (x - size // 2, y - road_width // 2, size, road_width))  # Horizontal road
        pygame.draw.rect(screen, GRAY, (x - road_width // 2, y - size // 2, road_width, size))  # Vertical road

        # Draw road markings
        line_width = 2
        line_length = 10
        line_gap = 10

        # Horizontal road markings
        for i in range(size // 2 // (line_length + line_gap)):
            # Left side
            pygame.draw.rect(screen, WHITE,
                             (x - size // 2 + i * (line_length + line_gap),
                              y - line_width // 2,
                              line_length, line_width))
            # Right side
            pygame.draw.rect(screen, WHITE,
                             (x + i * (line_length + line_gap),
                              y - line_width // 2,
                              line_length, line_width))

        # Vertical road markings
        for i in range(size // 2 // (line_length + line_gap)):
            # Top side
            pygame.draw.rect(screen, WHITE,
                             (x - line_width // 2,
                              y - size // 2 + i * (line_length + line_gap),
                              line_width, line_length))
            # Bottom side
            pygame.draw.rect(screen, WHITE,
                             (x - line_width // 2,
                              y + i * (line_length + line_gap),
                              line_width, line_length))

        # Draw intersection name
        text = font_medium.render(intersection_name, True, WHITE)
        screen.blit(text, (x - text.get_width() // 2, y - text.get_height() // 2))

        # Draw cycle time information
        cycle_time = cycle_times[intersection_name]
        offset = offset_times[intersection_name]
        adjusted_time = (time_in_cycle - offset) % cycle_time

        text_cycle = font_small.render(f"Cycle: {cycle_time}s", True, WHITE)
        text_current = font_small.render(f"Time: {adjusted_time:.1f}s", True, WHITE)
        text_offset = font_small.render(f"Offset: {offset}s", True, WHITE)

        screen.blit(text_cycle, (x - size // 2, y + size // 2 + 5))
        screen.blit(text_current, (x - size // 2, y + size // 2 + 25))
        screen.blit(text_offset, (x - size // 2, y + size // 2 + 45))

    # Draw traffic light
    def draw_traffic_light(position, road_id, is_green):
        x, y = position
        intersection, road = road_id.split("-")

        # Find position of light based on road
        if road == "1":  # Left light
            light_x = x - 20
            light_y = y
        elif road == "2":  # Top light
            light_x = x
            light_y = y - 20
        elif road == "3":  # Right light
            light_x = x + 20
            light_y = y
        elif road == "4":  # Bottom light
            light_x = x
            light_y = y + 20

        # Draw circle representing the light
        if is_green:
            pygame.draw.circle(screen, GREEN, (light_x, light_y), 8)
        else:
            pygame.draw.circle(screen, RED, (light_x, light_y), 8)

        # Draw road name
        text = font_small.render(road, True, WHITE)
        if road == "1":
            screen.blit(text, (light_x - 30, light_y - 10))
        elif road == "2":
            screen.blit(text, (light_x - 5, light_y - 30))
        elif road == "3":
            screen.blit(text, (light_x + 20, light_y - 10))
        elif road == "4":
            screen.blit(text, (light_x - 5, light_y + 15))

    # Draw connections between intersections
    def draw_connections():
        # Draw connections between intersections
        for intersection1 in network.intersections:
            for intersection2 in network.intersections:
                if intersection1 != intersection2 and network.connections[intersection1][intersection2]:
                    pos1 = intersection_positions[intersection1]
                    pos2 = intersection_positions[intersection2]

                    # Draw connection line
                    pygame.draw.line(screen, LIGHT_BLUE, pos1, pos2, 4)

                    # Calculate direction arrow
                    direction = (pos2[0] - pos1[0], pos2[1] - pos1[1])
                    length = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
                    normalized = (direction[0] / length, direction[1] / length)

                    # Midpoint of connection
                    mid_point = ((pos1[0] + pos2[0]) // 2, (pos1[1] + pos2[1]) // 2)

                    # Draw connection information
                    out_road, in_road = network.connections[intersection1][intersection2]
                    text = font_small.render(f"{out_road} -> {in_road}", True, WHITE)
                    screen.blit(text, mid_point)

    # Draw comparison table
    def draw_comparison_table():
        x, y = 20, 20
        width, height = 400, 150

        # Draw table background
        pygame.draw.rect(screen, (50, 50, 50), (x, y, width, height), border_radius=5)

        # Title
        text_title = font_medium.render("Optimization Model Comparison", True, WHITE)
        screen.blit(text_title, (x + 10, y + 10))

        # Cycle information
        independent_cycles = [cycle_times[intersection] for intersection in network.intersections]
        avg_cycle = sum(independent_cycles) / len(independent_cycles)
        max_cycle = max(independent_cycles)

        y_offset = 40
        text1 = font_small.render(f"Common cycle: {max_cycle}s", True, WHITE)
        text2 = font_small.render(f"Average cycle (if independent): {avg_cycle:.1f}s", True, WHITE)
        text3 = font_small.render(f"Improved with green wave and offset", True, WHITE)

        screen.blit(text1, (x + 10, y + y_offset))
        screen.blit(text2, (x + 10, y + y_offset + 25))
        screen.blit(text3, (x + 10, y + y_offset + 50))

        # Efficiency
        text4 = font_small.render("Total offset deviation: " +
                                  str(sum(offset_times.values())) + "s", True, WHITE)
        screen.blit(text4, (x + 10, y + y_offset + 75))

        text5 = font_small.render("Efficiency: Reduced stopping time between intersections", True, WHITE)
        screen.blit(text5, (x + 10, y + y_offset + 100))

    # Draw user instructions
    def draw_instructions():
        x, y = 750, 20
        width, height = 430, 120

        # Draw background
        pygame.draw.rect(screen, (50, 50, 50), (x, y, width, height), border_radius=5)

        # Title
        text_title = font_medium.render("User Instructions", True, WHITE)
        screen.blit(text_title, (x + 10, y + 10))

        # Control keys
        text1 = font_small.render("SPACE: Pause/Resume", True, WHITE)
        text2 = font_small.render("UP/DOWN: Increase/Decrease simulation speed", True, WHITE)
        text3 = font_small.render("R: Restart simulation", True, WHITE)
        text4 = font_small.render("ESC: Exit", True, WHITE)

        screen.blit(text1, (x + 10, y + 40))
        screen.blit(text2, (x + 10, y + 60))
        screen.blit(text3, (x + 10, y + 80))
        screen.blit(text4, (x + 10, y + 100))

    # Draw vehicles on connections
    def draw_vehicles_on_connections(simulation_time):
        for intersection1 in network.intersections:
            for intersection2 in network.intersections:
                if intersection1 != intersection2 and network.connections[intersection1][intersection2]:
                    pos1 = intersection_positions[intersection1]
                    pos2 = intersection_positions[intersection2]

                    # Get connection information
                    out_road, in_road = network.connections[intersection1][intersection2]
                    out_road_key = f"{intersection1}-{out_road}"
                    in_road_key = f"{intersection2}-{in_road}"

                    # Check if out_road has green light
                    out_road_signal_time = (simulation_time - offset_times[intersection1]) % cycle_times[intersection1]
                    is_out_road_green = False

                    # Determine phase of out_road
                    phase_with_out_road = None
                    for phase in network.phase_groups[intersection1]:
                        if out_road_key in phase:
                            phase_with_out_road = phase
                            break

                    # Calculate green light time for out_road
                    if phase_with_out_road == network.phase_groups[intersection1][0]:
                        # If it's phase 1
                        is_out_road_green = out_road_signal_time < signal_timings[intersection1][out_road_key][
                            "green_time"]
                    else:
                        # If it's phase 2
                        green_start = signal_timings[intersection1][network.phase_groups[intersection1][0][0]][
                                          "green_time"] + network.amber_time
                        is_out_road_green = green_start <= out_road_signal_time < green_start + \
                                            signal_timings[intersection1][out_road_key]["green_time"]

                    # Only draw vehicles when light is green
                    if is_out_road_green:
                        # Calculate direction
                        direction = (pos2[0] - pos1[0], pos2[1] - pos1[1])
                        length = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
                        normalized = (direction[0] / length, direction[1] / length)

                        # Calculate number of vehicles based on congestion
                        congestion = congestion_data[out_road_key]
                        num_vehicles = int(congestion * 15)

                        # Draw vehicles on the connection
                        for k in range(num_vehicles):
                            # Calculate vehicle position based on time
                            progress = (simulation_time * speed_factor / 10 + k / num_vehicles) % 1
                            vehicle_x = pos1[0] + progress * direction[0]
                            vehicle_y = pos1[1] + progress * direction[1]

                            # Draw rectangle representing vehicle
                            vehicle_size = 8
                            pygame.draw.rect(screen, YELLOW,
                                             (vehicle_x - vehicle_size // 2,
                                              vehicle_y - vehicle_size // 2,
                                              vehicle_size, vehicle_size))

    # Main loop
    running = True
    simulation_time = 0
    paused = False
    speed_factor = 1
    max_cycle_time = max(cycle_times.values())

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    speed_factor = min(speed_factor + 0.5, 3)
                elif event.key == pygame.K_DOWN:
                    speed_factor = max(speed_factor - 0.5, 0.5)
                elif event.key == pygame.K_r:
                    simulation_time = 0
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # Update simulation time
        if not paused:
            simulation_time = (simulation_time + 0.1 * speed_factor) % max_cycle_time

        # Clear screen
        screen.fill((30, 30, 30))

        # Draw connections between intersections
        draw_connections()

        # Draw vehicles moving between intersections
        draw_vehicles_on_connections(simulation_time)

        # Draw intersections
        for intersection_name, position in intersection_positions.items():
            # Calculate current time for this intersection (including offset)
            intersection_time = (simulation_time - offset_times[intersection_name]) % cycle_times[intersection_name]

            # Draw intersection
            draw_intersection(position, intersection_size, intersection_name, intersection_time)

            # Determine and draw light status
            for road in network.roads_per_intersection:
                road_id = f"{intersection_name}-{road}"

                # Check if light is green
                is_green = False

                # Find phase for this road
                for phase_index, phase in enumerate(network.phase_groups[intersection_name]):
                    if road_id in phase:
                        # Calculate start and end time of phase
                        if phase_index == 0:
                            green_start = 0
                        else:
                            green_start = \
                            signal_timings[intersection_name][network.phase_groups[intersection_name][0][0]][
                                "green_time"] + network.amber_time

                        green_end = green_start + signal_timings[intersection_name][road_id]["green_time"]

                        # Check if current time is during green light
                        if green_start <= intersection_time < green_end:
                            is_green = True
                            break

                # Draw traffic light
                draw_traffic_light(traffic_light_positions[road_id], road_id, is_green)

        # Draw information and instructions
        draw_comparison_table()
        draw_instructions()

        # Draw simulation information
        text_time = font_medium.render(f"Simulation time: {simulation_time:.1f}s", True, WHITE)
        text_speed = font_medium.render(f"Speed: x{speed_factor:.1f}", True, WHITE)
        text_status = font_medium.render("Status: " + ("Paused" if paused else "Running"), True, WHITE)

        screen.blit(text_time, (20, 200))
        screen.blit(text_speed, (20, 230))
        screen.blit(text_status, (20, 260))

        # Update display
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main():
    print("=== SMART TRAFFIC LIGHT NETWORK CONTROL SYSTEM ===")
    print("Using Webster's algorithm combined with Fuzzy Logic and network synchronization")
    print("-----------------------------------------------------------")

    try:
        # Predefined congestion data sets
        predefined_data_sets = {
            "1": {  # Uniform density case
                "A-1": 0.3, "A-2": 0.3, "A-3": 0.3, "A-4": 0.3,
                "B-1": 0.3, "B-2": 0.3, "B-3": 0.3, "B-4": 0.3,
                "C-1": 0.3, "C-2": 0.3, "C-3": 0.3, "C-4": 0.3,
                "D-1": 0.3, "D-2": 0.3, "D-3": 0.3, "D-4": 0.3,
                "E-1": 0.3, "E-2": 0.3, "E-3": 0.3, "E-4": 0.3,
                "F-1": 0.3, "F-2": 0.3, "F-3": 0.3, "F-4": 0.3
            },
            "2": {  # Increasing density from A to F
                "A-1": 0.2, "A-2": 0.2, "A-3": 0.2, "A-4": 0.2,
                "B-1": 0.3, "B-2": 0.3, "B-3": 0.3, "B-4": 0.3,
                "C-1": 0.4, "C-2": 0.4, "C-3": 0.4, "C-4": 0.4,
                "D-1": 0.5, "D-2": 0.5, "D-3": 0.5, "D-4": 0.5,
                "E-1": 0.6, "E-2": 0.6, "E-3": 0.6, "E-4": 0.6,
                "F-1": 0.7, "F-2": 0.7, "F-3": 0.7, "F-4": 0.7
            },
            "3": {  # Horizontal roads (1-3) busier than vertical roads (2-4)
                "A-1": 0.7, "A-2": 0.3, "A-3": 0.7, "A-4": 0.3,
                "B-1": 0.7, "B-2": 0.3, "B-3": 0.7, "B-4": 0.3,
                "C-1": 0.7, "C-2": 0.3, "C-3": 0.7, "C-4": 0.3,
                "D-1": 0.7, "D-2": 0.3, "D-3": 0.7, "D-4": 0.3,
                "E-1": 0.7, "E-2": 0.3, "E-3": 0.7, "E-4": 0.3,
                "F-1": 0.7, "F-2": 0.3, "F-3": 0.7, "F-4": 0.3
            },
            "4": {  # Vertical roads (2-4) busier than horizontal roads (1-3)
                "A-1": 0.3, "A-2": 0.7, "A-3": 0.3, "A-4": 0.7,
                "B-1": 0.3, "B-2": 0.7, "B-3": 0.3, "B-4": 0.7,
                "C-1": 0.3, "C-2": 0.7, "C-3": 0.3, "C-4": 0.7,
                "D-1": 0.3, "D-2": 0.7, "D-3": 0.3, "D-4": 0.7,
                "E-1": 0.3, "E-2": 0.7, "E-3": 0.3, "E-4": 0.7,
                "F-1": 0.3, "F-2": 0.7, "F-3": 0.3, "F-4": 0.7
            },
            "5": {  # One busiest intersection (Node C)
                "A-1": 0.3, "A-2": 0.3, "A-3": 0.3, "A-4": 0.3,
                "B-1": 0.3, "B-2": 0.3, "B-3": 0.3, "B-4": 0.3,
                "C-1": 0.8, "C-2": 0.8, "C-3": 0.8, "C-4": 0.8,
                "D-1": 0.3, "D-2": 0.3, "D-3": 0.3, "D-4": 0.3,
                "E-1": 0.3, "E-2": 0.3, "E-3": 0.3, "E-4": 0.3,
                "F-1": 0.3, "F-2": 0.3, "F-3": 0.3, "F-4": 0.3
            }
        }

        print("\nChoose how to input traffic density data:")
        print("1. Use predefined data set")
        print("2. Enter manually")
        data_choice = input("Your choice (1/2): ")

        congestion_data = {}

        if data_choice == "1":
            print("\nSelect traffic density data set:")
            print("1. Uniform density (0.3)")
            print("2. Increasing density from A to F")
            print("3. Horizontal roads (1-3) busier than vertical roads (2-4)")
            print("4. Vertical roads (2-4) busier than horizontal roads (1-3)")
            print("5. One busiest intersection (Node C)")

            data_set_choice = input("Select data set (1-5): ")

            if data_set_choice in predefined_data_sets:
                congestion_data = predefined_data_sets[data_set_choice]
                print("\nSelected congestion index data:")
                for node in "ABCDEF":
                    print(f"Node {node}:", end=" ")
                    for road in range(1, 5):
                        print(f"{road}={congestion_data[f'{node}-{road}']:.2f}", end=" ")
                    print()
            else:
                print("Invalid choice! Using default data set (1).")
                congestion_data = predefined_data_sets["1"]

        elif data_choice == "2":
            print("\nEnter congestion index (0.01-1.00) for each road at each intersection:")
            for node in "ABCDEF":
                print(f"\nIntersection {node}:")
                for road in range(1, 5):
                    value = float(input(f"  Congestion index for road {road}: "))
                    if not (0.01 <= value <= 1.00):
                        print("Error: Congestion index must be between 0.01 and 1.00")
                        return
                    congestion_data[f"{node}-{road}"] = value

            # Display entered information
            print("\nEntered congestion index data:")
            for node in "ABCDEF":
                print(f"Node {node}:", end=" ")
                for road in range(1, 5):
                    print(f"{road}={congestion_data[f'{node}-{road}']:.2f}", end=" ")
                print()

        else:
            print("Invalid choice! Using default data set (1).")
            congestion_data = predefined_data_sets["1"]

        # Create traffic network
        network = TrafficNetwork(congestion_data)

        # Ask user which optimization mode to use
        print("\nSelect optimization method:")
        print("1. Independent optimization (each intersection optimized separately)")
        print("2. Network optimization (entire network optimized simultaneously)")
        choice = input("Your choice (1/2): ")

        if choice == "1":
            signal_timings, cycle_times = network.optimize_independent()
            network.offset_times = {intersection: 0 for intersection in network.intersections}

            # Display results
            print("\nIndependent optimization results:")
            for intersection in network.intersections:
                print(f"\n{intersection}:")
                print(f"Cycle: {cycle_times[intersection]} seconds")
                for road in network.roads_per_intersection:
                    road_id = f"{intersection}-{road}"
                    timing = signal_timings[intersection][road_id]
                    print(
                        f"  Road {road}: Green={timing['green_time']}s, Amber={timing['amber_time']}s, Red={timing['red_time']}s")

        elif choice == "2":
            signal_timings, cycle_times, offset_times = network.optimize_network()

            # Display results
            print("\nNetwork optimization results:")
            for intersection in network.intersections:
                print(f"\n{intersection}:")
                print(f"Cycle: {cycle_times[intersection]} seconds")
                print(f"Offset: {offset_times[intersection]} seconds")
                for road in network.roads_per_intersection:
                    road_id = f"{intersection}-{road}"
                    timing = signal_timings[intersection][road_id]
                    print(
                        f"  Road {road}: Green={timing['green_time']}s, Amber={timing['amber_time']}s, Red={timing['red_time']}s")

        else:
            print("Invalid choice!")
            return

        # Ask user if they want to run the simulation
        print("\nDo you want to run the simulation? (y/n)")
        if input().lower() == 'y':
            print("Starting simulation...")
            try:
                run_network_simulation(network)
            except Exception as e:
                print(f"Error running simulation: {e}")
                print("Make sure you have installed pygame: pip install pygame")

    except ValueError:
        print("Error: Please enter valid numeric values")
    except ImportError as e:
        print(f"Error: {e}")
        print("You need to install these libraries to run the program:")
        print("pip install numpy scikit-fuzzy pygame")


if __name__ == "__main__":
    main()