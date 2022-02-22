import pandas as pd
import json
from requests import get, put
from utils.conversion import conversion
from time import sleep


class icasa:
    def __init__(self):
        self.root = f'http://localhost:9000'

        self.get_devices_url = f'{self.root}/icasa/devices/devices'
        self.get_zones_url = f'{self.root}/icasa/zones/zones'
        self.get_persons_url = f'{self.root}/icasa/persons/persons'

        self.put_device_url = f'{self.root}/icasa/devices/device'
        self.put_zone_url = f'{self.root}/icasa/zones/zone'
        self.put_person_url = f'{self.root}/icasa/persons/person'

        self.speed_url = f'{self.root}/icasa/clocks/clock/default'
        self.restart_url = f'{self.root}/icasaRestart'

        self.data = None

        self.read_kind = {"T": 'device',
                          "S": 'device',
                          "W": 'device',
                          "CO2": 'device',
                          "L": 'device',
                          "H": 'device',
                          "C": 'device',
                          "O": 'device',
                          "A": 'device',
                          "Pr": 'device',
                          "Jessica": 'person',
                          "room": 'zone',
                          "outdoor": 'zone'
                        }

        self.read_devices = {"Thermometer-b1f74267ed": 'thermometer.currentTemperature',
                             "ToggleSwitch-d9704c5a0f": 'powerSwitch.currentStatus',
                             "DoorWindowSensor-b041e06747": 'doorWindowSensor.opneningDetection',
                             'COGasSensor-1bc0572e9e': 'carbonMonoxydeSensor.currentConcentration',
                             "PushButton-5ec148b252": 'pushButton.pushAndHold',
                             "CO2GasSensor-b66d761212": 'carbonDioxydeSensor.currentConcentration',
                             "BinaryLight-5022136575": 'binaryLight.powerStatus',
                             "Heater-1bd6fdc99a": 'heater.powerLevel',
                             "Cooler-ce2209b064": 'cooler.powerLevel',
                             "Thermometer-829cc07927": 'thermometer.currentTemperature',
                             "Siren-d6c226252a": 'siren.status',
                             "PresenceSensor-cd52a5fed8": 'presenceSensor.sensedPresence'
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

    # GET a new sample
    def sample(self):
        new_data = {}

        # Read sensor data
        resp_device = get(self.get_devices_url)
        if resp_device.status_code != 200:
            print("data loading fail!")

        for device in resp_device.json():
            id = device['id']
            if id not in self.read_devices:
                print('Device not known!')
                continue
            for param in device['properties']:
                if param['name'] == self.read_devices[id]:
                    new_data[[k for k, v in self.read_devices_short.items() if v == id][0]] = [param['value']]

        n_data = pd.DataFrame(new_data)
        return n_data

    # Intervention: PUT request
    def intervention(self, evidence, resp_time=0):
        for kind, (name, property), value in evidence:
            if kind == 'device':
                resp = put(self.put_device_url + f'/{name}',
                                    data=json.dumps({'id': name, property: value}))
            elif kind == 'zone':
                resp = put(self.put_zone_url + f'/{name}',
                                    data=json.dumps({property: value}))
            elif kind == 'person':
                resp = put(self.put_device_url + f'/{name}',
                           data=json.dumps({'id': name, property: value}))
            else:
                assert False, 'not recognize option: kind'

            if resp.status_code != 200:
                print("pushing data fail!", "Error code: ", resp.status_code)

        print('waiting for response...', end='')
        sleep(resp_time) if resp_time > 0 else None
        print('[OK]')

        return self.sample()

    def formatEvidence(self, evidence):
        return [(self.read_kind[name], (self.read_devices_short[name], self.read_devices[self.read_devices_short[name]]),
                 value) for name, value in evidence.items()]

    def do(self, evidence, do_size=2, resp_time=2):
        if evidence is None:
            evidence = {}

        # Format evidence
        evidence = self.formatEvidence(evidence)

        column = [f'{k}' for k, v in self.read_devices_short.items()]
        # Append Pow column
        column.append('Pow')

        # Initialize dataset
        df = pd.DataFrame(columns={node: [] for node in column}).astype(int)

        # Make interventions
        for i in range(do_size):
            new_sample = self.intervention(evidence, resp_time=resp_time)

            # Manually adding Pow value since it is not a device
            light = 100 if new_sample['L'].bool() else 0
            new_sample['Pow'] = light + new_sample['H'] + new_sample['C']

            df = pd.concat([df, conversion(new_sample)], axis=0)

        df.reset_index(drop=True, inplace=True)

        return df


if __name__ == '__main__':
    home = icasa()

    # TEST sample, status: working
    # ret = home.sample()
    # print(ret)
    # import warnings
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     print(conversion(ret))

    # TEST intervention, status: working
    # evidence = ['BinaryLight-5022136575', 'binaryLight.powerStatus', 'true']
    # ret = home.intervention(evidence)
    # print(ret)

    # TEST do, status: working
    evidence = {'O': 300.0}
    ret = home.do(evidence)
    print(ret)




