import pandas as pd
import json
import requests
from time import sleep
import numpy as np


class iCasa:
    def __init__(self, instance=None):
        self.__root = f'http://localhost:9000'

        # self.__root = f'https://explainableai.fr'
        # self.__auth = HTTPBasicAuth('reader', 'xai')

        if instance is not None:
            self.__root += f'/{instance}'

        self.__get_device_url = f'{self.__root}/icasa/devices/devices'
        self.__get_zone_url = f'{self.__root}/icasa/zones/zones'

        self.__put_device_url = f'{self.__root}/icasa/devices/device'
        self.__put_zone_url = f'{self.__root}/icasa/zones/zone'

        self.__speed_url = f'{self.__root}/icasa/clocks/clock/default'
        self.__restart_url = f'{self.__root}/icasaRestart'

        self.set_devices = {"temperature_controller": "m_target_temperature",
                            'window': 'm_open',
                            'window': 'm_blinds_open',
                            'heater': 'm_heater.powerLevel'}

        self.set_zones = {'outdoor': 'myTemperature'}

        # self.read_devices = {"heater": ['heater.powerLevel', 'm_heater.powerLevel'],
        #                      "thermometer": ['etienne.temperature'],
        #                      "out_thermometer": ['etienne.temperature'],
        #                      'window': ['blinds_open', 'window.opened', 'm_open', 'm_blinds_open'],
        #                      "temperature_controller": ['m_target_temperature', 'state'],
        #                      }

        self.controlable = ['window__m_open', 'window__m_blinds_open'] \
                           + ["heater__m_heater.powerLevel", "out_thermometer__etienne.temperature",
                              'temperature_controller__state', 'temperature_controller__m_target_temperature']

        self.data = None

        self.name2kind = {'outdoor': 'zone', 'window': 'device', 'out_thermometer': 'device',
                          'heater': 'device', 'temperature_controller': 'device', 'Thermometer-b1f74267ed': 'device'}

        self.name = lambda k, p, v: (self.name2kind[k], (k, p), float(v))

        self.state = [('window', 'm_open'), ('window', 'blinds_open'), ('heater', 'm_heater.powerLevel'),
                      ('temperature_controller', 'state')]
        self.temperature = [0, 7, 17, 23, 28, 30]
        self.reset_auto()
        self._do([self.name('temperature_controller', 'm_target_temperature', 20)], resp_time=0)

        self.read_devices = {"Thermometer-b1f74267ed": ['thermometer.currentTemperature'],
                             "ToggleSwitch-d9704c5a0f": ['powerSwitch.currentStatus'],
                             "DoorWindowSensor-b041e06747": ['doorWindowSensor.opneningDetection'],
                             'COGasSensor-1bc0572e9e': ['carbonMonoxydeSensor.currentConcentration'],
                             "PushButton-5ec148b252": ['pushButton.pushAndHold'],
                             "CO2GasSensor-b66d761212": ['carbonDioxydeSensor.currentConcentration'],
                             "BinaryLight-5022136575": ['binaryLight.powerStatus'],
                             "Heater-1bd6fdc99a": ['heater.powerLevel'],
                             "Cooler-ce2209b064": ['cooler.powerLevel'],
                             "Thermometer-829cc07927": ['thermometer.currentTemperature'],
                             "Siren-d6c226252a": ['siren.status'],
                             "PresenceSensor-cd52a5fed8": ['presenceSensor.sensedPresence']
                             }

        self.read_devices_short = {
            "S": "ToggleSwitch-d9704c5a0f",
            "W": "DoorWindowSensor-b041e06747",
            "CO": "COGasSensor-1bc0572e9e",
            "B": "PushButton-5ec148b252",
            "CO2": "CO2GasSensor-b66d761212",
            "L": "BinaryLight-5022136575",
            "H": "Heater-1bd6fdc99a",
            "C": "Cooler-ce2209b064",
            "O": "Thermometer-829cc07927",
            "A": "Siren-d6c226252a",
            "Pr": "PresenceSensor-cd52a5fed8",
            "T": "Thermometer-b1f74267ed"
        }

    def acceleration(self):
        resp = requests.get(self.__speed_url)  # , auth=self.__auth)
        if resp.status_code != 200:
            print("acceleration failed!")
            return
        param = resp.json()
        print(param)
        param['factor'] = 100
        requests.put(self.__speed_url,
                     data=json.dumps(param))  # , auth=self.__auth)

        print('acceleration speed up !')

    # Sampling: GET request
    def fetch_data(self) -> pd.DataFrame:
        new_data = {}

        # Read sensor data
        resp_device = requests.get(self.__get_device_url)  # , auth=self.__auth)
        if resp_device.status_code != 200:
            print("data loading fail!")

        for device in resp_device.json():
            id = device['id']
            if id not in self.read_devices:
                continue
            for param in device['properties']:
                if param['name'] in self.read_devices[id]:
                    new_data[[k for k, v in self.read_devices_short.items() if v == id][0]] = [param['value']]

        # update with new sensor data
        n_data = pd.DataFrame(new_data)
        return n_data

    # Probably each time that we do a sample, this function adds it to data, in order to build an observational dataset
    def add_record(self) -> pd.DataFrame:
        new_data = self.fetch_data()
        if self.data is None:
            self.data = new_data
        else:
            self.data = pd.concat((self.data, new_data))
        return self.data

    def reset_auto(self):
        T = np.random.choice(self.temperature)
        to_do = [(self.name(remain[0], remain[1], -1)) for remain in self.state] \
                + [('zone', ('outdoor', 'myTemperature'), float(T))]
        self._do(to_do, resp_time=0)

    # Intervention: PUT request
    def _do(self, evidence, resp_time=0, verbose=False) -> pd.DataFrame:
        """
        evidence: list of (kind, name, val)
        resp_time : (in sec) waiting time before fetching data
        """
        for kind, (name, property), value in evidence:
            if kind == 'device':
                resp = requests.put(self.__put_device_url + f'/{name}',
                                    data=json.dumps({'id': name, property: value}))
            elif kind == 'zone':
                resp = requests.put(self.__put_zone_url + f'/{name}',
                                    data=json.dumps({property: value}))
            else:
                assert False, 'not recognize option: kind'

            if resp.status_code != 200:
                print("pushing data fail!", "Error code: ", resp.status_code)

        print('waiting for response...', end='') if verbose else None
        sleep(resp_time) if resp_time > 0 else None
        print('[OK]') if verbose else None

        return self.fetch_data()

    # ho cambiato la size per motivi di testing
    def do(self, evidence, size=2, seed=12, resp_time=3, verbose=True):
        np.random.seed(seed)

        if evidence is None:
            evidence = {}

        # When we have the following format: evidence = {'T': 0}
        evidence_ = [("device", (self.read_devices_short[name], self.read_devices[self.read_devices_short[name]][0]), value) for
                     name, value in evidence.items()]
        column = [f'{k}' for k, v in self.read_devices_short.items()]
        # Remember to insert Pow

        # When we have the following format: evidence = {'Thermometer-b1f74267ed__thermometer.currentTemperature': 0}
        # evidence_ = [self.name(k.split('__')[0], k.split('__')[1], v) for k, v in evidence.items()]
        # column = [f'{k}__{v[0]}' for k, v in self.read_devices.items()]

        df = pd.DataFrame({node: [] for node in column}).astype(int)
        for i in range(size):
            # Conversione in booleano prima di concatenare: probabilmente le soglie sono da ricalibrare
            # df = pd.concat(df, conversion(self._do(evidence_, resp_time=resp_time, verbose=True)))
            new_sample = self._do(evidence_, resp_time=resp_time, verbose=True)
            df = pd.concat([df, new_sample], axis=0)
            # self.reset_auto()

        df.reset_index(drop=True, inplace=True)

        return df


if __name__ == '__main__':
    # home = iCasa('icasaInterface')

    home = iCasa()

    # fix = [('device', ('Thermometer-b1f74267ed', 'thermometer.currentTemperature'), 400)]
    # print(home._do(fix).columns)

    # to_do = {'Thermometer-b1f74267ed__thermometer.currentTemperature': 0}

    # to_do = {'H': 1000}
    # ret = home.do(to_do)
    # print(ret)

    ret = home.fetch_data()
    print(ret)

    # Sembra che l'intervention non riesca ad avere effetto sui dispositivi, cioè la PUT ritorna 200, ma il valore
    # del dispositivo su iCasa non si modifica

    # home.fetch_data().to_csv('C:\\Users\\pakyr\\Desktop\\tmp.csv', sep=',', index=False)

    # Il problema è che dal codice del learning la evidence arriva in questo formato
    # evidence = {'P': 0}
    # mentre qui la do la vuole nel formato
    # kind, (name, property), value
    # quindi bisogna fare una conversione.
    # L'ulteriore problema è successivamente quando si va a controllare la lista dei dispositivi sui quali intervenire/leggere
    # Le API funzionano con gli id del dispositivo/zona quindi o si cambiano nel codice oppure nello script iCasa



    # home.acceleration()
    # state = [('window', 'm_open'), ('window', 'm_blinds_open'), ('heater', 'm_heater.powerLevel')]
    #
    # n_data = 1000
    # np.random.seed(0)
    # temperature = [0, 7, 17, 23, 28, 30]
    #
    # for _ in tqdm(range(n_data)):
    #     T = np.random.choice(temperature)
    #     # select random one with random value and set the rest in auto mode
    #     state_ = state.copy()
    #     np.random.shuffle(state_)
    #     select = state_.pop()
    #     val = np.random.choice([0, 1, -1])
    #     to_do = [(home.name(select[0], select[1], val))] \
    #             + [(home.name(remain[0], remain[1], -1)) for remain in state_] \
    #             + [('zone', ('outdoor', 'myTemperature'), float(T))]
    #
    #     resp = home._do(to_do, resp_time=3)
    #
    #     home.add_record().to_csv('ocik/demo/store/icasa2.csv', index=False)
    #
    # home.data.to_csv('ocik/demo/store/icasa2.csv', index=False)
