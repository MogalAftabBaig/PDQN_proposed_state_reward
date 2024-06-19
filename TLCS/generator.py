import numpy as np
import math
import random

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("intersection/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
              <vType accel="1.0" deccel="4.5" id="Car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5"/>
              <vType accel="1.0" deccel="5.0" id="Bus" length="12.0" maxSpeed="10" sigma="0.0"/>
              <route id="r1" edges="-h11 -h12 -h13"/>
              <route id="r2" edges="-v11 -v12 -v13"/>
              <route id="r3" edges="-h21 -h22 -h23"/>
              <route id="r4" edges="-v21 -v22 -v23"/>
              <route id="r5" edges="-h11 v11"/>
              <route id="r6" edges="-h21 v12 v11"/>
              <route id="r7" edges="-h11 -v12 -v13"/>
              <route id="r8" edges="-h21 -v13"/>
              <route id="r9" edges="h13 h12 h11"/>
              <route id="r10" edges="v13 v12 v11"/>
              <route id="r11" edges="h23 h22 h21"/>
              <route id="r12" edges="v23 v22 v21"/>
              <route id="r13" edges="-v11 -h12 -h13"/>
              <route id="r14" edges="v13 -h22 -h23"/>
              <route id="r15" edges="v13 h21"/>
              <route id="r16" edges="v23 -h23"/>""", file=routes)

            for vehNr, i in enumerate(car_gen_steps):
                num=random.randint(1,16)
                if num==1:
                    print(' <vehicle id="v_%i" type="Car" route="r1" depart="%i" />' % (vehNr, i), file=routes)
                elif num==2:
                    print(' <vehicle id="v_%i" type="Car" route="r2" depart="%i" />' % (vehNr, i), file=routes)
                elif num==3:
                    print(' <vehicle id="v_%i" type="Car" route="r3" depart="%i" />' % (vehNr, i), file=routes)
                elif num==4:
                    print(' <vehicle id="v_%i" type="Car" route="r4" depart="%i" />' % (vehNr, i), file=routes)
                elif num==5:
                    print(' <vehicle id="v_%i" type="Car" route="r5" depart="%i" />' % (vehNr, i), file=routes)
                elif num==6:
                    print(' <vehicle id="v_%i" type="Car" route="r6" depart="%i" />' % (vehNr, i), file=routes)
                elif num==7:
                    print(' <vehicle id="v_%i" type="Car" route="r7" depart="%i" />' % (vehNr, i), file=routes)
                elif num==8:
                    print(' <vehicle id="v_%i" type="Car" route="r8" depart="%i" />' % (vehNr, i), file=routes)
                elif num==9:
                    print(' <vehicle id="v_%i" type="Car" route="r9" depart="%i" />' % (vehNr, i), file=routes)
                elif num==10:
                    print(' <vehicle id="v_%i" type="Car" route="r10" depart="%i" />' % (vehNr, i), file=routes)
                elif num==11:
                    print(' <vehicle id="v_%i" type="Car" route="r11" depart="%i" />' % (vehNr, i), file=routes)
                elif num==12:
                    print(' <vehicle id="v_%i" type="Car" route="r12" depart="%i" />' % (vehNr, i), file=routes)
                elif num==13:
                    print(' <vehicle id="v_%i" type="Car" route="r13" depart="%i" />' % (vehNr, i), file=routes)
                elif num==14:
                    print(' <vehicle id="v_%i" type="Car" route="r14" depart="%i" />' % (vehNr, i), file=routes)
                elif num==15:
                    print(' <vehicle id="v_%i" type="Car" route="r15" depart="%i" />' % (vehNr, i), file=routes)
                else:
                    print(' <vehicle id="v_%i" type="Car" route="r16" depart="%i" />' % (vehNr, i), file=routes)

            print("</routes>", file=routes)

    def generate_routefile_normal(self, seed):
        """
        Generation of the route of every car for one episode using a normal distribution.
        """
        np.random.seed(seed)  # make tests reproducible
        # the generation of cars is distributed according to a normal distribution
        timings = np.random.normal(loc=self._max_steps / 2, scale=self._max_steps / 10, size=self._n_cars_generated)
        timings = np.clip(timings, 0, self._max_steps)  # clip to ensure values are within the desired range
        timings = np.sort(timings)
        car_gen_steps = np.rint(timings)  # round every value to int -> effective steps when a car will be generated
        # produce the file for cars generation, one car per line
        with open("intersection/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
              <vType accel="1.0" deccel="4.5" id="Car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5"/>
              <vType accel="1.0" deccel="5.0" id="Bus" length="12.0" maxSpeed="10" sigma="0.0"/>
              <route id="r1" edges="-h11 -h12 -h13"/>
              <route id="r2" edges="-v11 -v12 -v13"/>
              <route id="r3" edges="-h21 -h22 -h23"/>
              <route id="r4" edges="-v21 -v22 -v23"/>
              <route id="r5" edges="-h11 v11"/>
              <route id="r6" edges="-h21 v12 v11"/>
              <route id="r7" edges="-h11 -v12 -v13"/>
              <route id="r8" edges="-h21 -v13"/>
              <route id="r9" edges="h13 h12 h11"/>
              <route id="r10" edges="v13 v12 v11"/>
              <route id="r11" edges="h23 h22 h21"/>
              <route id="r12" edges="v23 v22 v21"/>
              <route id="r13" edges="-v11 -h12 -h13"/>
              <route id="r14" edges="v13 -h22 -h23"/>
              <route id="r15" edges="v13 h21"/>
              <route id="r16" edges="v23 -h23"/>""", file=routes)

            for vehNr, i in enumerate(car_gen_steps):
                num=random.randint(1,16)
                if num==1:
                    print(' <vehicle id="v_%i" type="Car" route="r1" depart="%i" />' % (vehNr, i), file=routes)
                elif num==2:
                    print(' <vehicle id="v_%i" type="Car" route="r2" depart="%i" />' % (vehNr, i), file=routes)
                elif num==3:
                    print(' <vehicle id="v_%i" type="Car" route="r3" depart="%i" />' % (vehNr, i), file=routes)
                elif num==4:
                    print(' <vehicle id="v_%i" type="Car" route="r4" depart="%i" />' % (vehNr, i), file=routes)
                elif num==5:
                    print(' <vehicle id="v_%i" type="Car" route="r5" depart="%i" />' % (vehNr, i), file=routes)
                elif num==6:
                    print(' <vehicle id="v_%i" type="Car" route="r6" depart="%i" />' % (vehNr, i), file=routes)
                elif num==7:
                    print(' <vehicle id="v_%i" type="Car" route="r7" depart="%i" />' % (vehNr, i), file=routes)
                elif num==8:
                    print(' <vehicle id="v_%i" type="Car" route="r8" depart="%i" />' % (vehNr, i), file=routes)
                elif num==9:
                    print(' <vehicle id="v_%i" type="Car" route="r9" depart="%i" />' % (vehNr, i), file=routes)
                elif num==10:
                    print(' <vehicle id="v_%i" type="Car" route="r10" depart="%i" />' % (vehNr, i), file=routes)
                elif num==11:
                    print(' <vehicle id="v_%i" type="Car" route="r11" depart="%i" />' % (vehNr, i), file=routes)
                elif num==12:
                    print(' <vehicle id="v_%i" type="Car" route="r12" depart="%i" />' % (vehNr, i), file=routes)
                elif num==13:
                    print(' <vehicle id="v_%i" type="Car" route="r13" depart="%i" />' % (vehNr, i), file=routes)
                elif num==14:
                    print(' <vehicle id="v_%i" type="Car" route="r14" depart="%i" />' % (vehNr, i), file=routes)
                elif num==15:
                    print(' <vehicle id="v_%i" type="Car" route="r15" depart="%i" />' % (vehNr, i), file=routes)
                else:
                    print(' <vehicle id="v_%i" type="Car" route="r16" depart="%i" />' % (vehNr, i), file=routes)

            print("</routes>", file=routes)
