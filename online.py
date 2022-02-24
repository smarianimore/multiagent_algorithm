import pandas as pd
import json
from requests import get, put
from utils.conversion import conversion
from time import sleep
from utils.conversion import evidence_to_numeric
from utils.config import read_kind, read_devices, read_devices_short, HOST, PORT
import socket

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

        self.read_kind = read_kind
        self.read_devices = read_devices
        self.read_devices_short = read_devices_short

        self.host = HOST
        self.port = PORT

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

        # Manually adding Pow value since it is not a device
        light = 100 if n_data['L'].bool() else 0
        n_data['Pow'] = light + n_data['H'] + n_data['C']

        return n_data

    # The function actualizes the device values with a put request
    # def intervention(self, evidence, resp_time=0):
    #     for kind, (name, property), value in evidence:
    #         if kind == 'device':
    #             resp = put(self.put_device_url + f'/{name}',
    #                                 data=json.dumps({'id': name, property: value}))
    #         elif kind == 'zone':
    #             resp = put(self.put_zone_url + f'/{name}',
    #                                 data=json.dumps({property: value}))
    #         elif kind == 'person':
    #             resp = put(self.put_device_url + f'/{name}',
    #                        data=json.dumps({'id': name, property: value}))
    #         else:
    #             assert False, 'not recognize option: kind'
    #
    #         if resp.status_code != 200:
    #             print("pushing data fail!", "Error code: ", resp.status_code)
    #
    #     print('waiting for response...', end='')
    #     sleep(resp_time) if resp_time > 0 else None
    #     print('[OK]')
    #
    #     return self.sample()

    def intervention(self, evidence, resp_time=0):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))

        evidence = str(evidence) + "\n"
        evidence = str.encode(evidence)
        sock.sendall(evidence)

        print('waiting for response...', end='')
        sleep(resp_time) if resp_time > 0 else None
        print('[OK]')


    def format_evidence(self, evidence):
        return [(self.read_kind[name], (self.read_devices_short[name], self.read_devices[self.read_devices_short[name]]),
                 value) for name, value in evidence.items()]

    def do(self, evidence, do_size=2, resp_time=2):
        if evidence is None:
            evidence = {}

        # Format evidence
        evidence = self.format_evidence(evidence_to_numeric(evidence))

        column = [f'{k}' for k, v in self.read_devices_short.items()]
        # Append Pow column
        column.append('Pow')

        # Initialize dataset
        df = pd.DataFrame(columns={node: [] for node in column}).astype(int)

        # Intervention
        self.intervention(evidence)

        # Make interventions
        for i in range(do_size):
            new_sample = self.sample()
            df = pd.concat([df, conversion(new_sample)], axis=0)
            sleep(resp_time) if resp_time > 0 else None

        df.reset_index(drop=True, inplace=True)

        return df

    # The function simulates the values without making a real intervention
    def simulate(self, evidence, do_size=2):

        evidence = evidence_to_numeric(evidence)

        node = str(tuple(evidence.items())[0][0])
        value = tuple(evidence.items())[0][1]

        df = pd.DataFrame(columns=['Pr', 'L', 'T', 'W', 'H', 'Pow']).astype(int)

        sample = {}
        for i in range(do_size):

            # Simulate past values
            if i == 0:
                for n in df.columns:
                    if n == 'Pr':
                        sample['Pr'] = 0
                    elif n == 'L':
                        sample['L'] = 0
                    elif n == 'Pow':
                        sample['Pow'] = 500
                    elif n == 'H':
                        sample['H'] = 600
                    elif n == 'W':
                        sample['W'] = 0
                    elif n == 'T':
                        sample['T'] = 293.15

                # Intervention
                sample[node] = value

            # Change values based on intervention
            if node == 'L':
                sample['Pow'] = sample['Pow'] * 2 if sample['L'] == 1 else sample['Pow'] // 2
            elif node == 'H':
                sample['Pow'] = sample['Pow'] * 2 if sample['H'] > 500 else sample['Pow'] // 2
                sample['T'] = sample['T'] * 2 if sample['H'] > 500 else sample['T'] // 2
            elif node == 'W':
                sample['T'] = sample['T'] * 2 if sample['W'] == 0 else sample['T'] // 2

            # Add sample to dataset
            df = df.append(pd.Series(sample, name=i))

        return conversion(df)


if __name__ == '__main__':
    home = icasa()

    # TEST sample, status: working
    # ret = home.sample()
    # print(ret)

    # TEST intervention, status: working
    # evidence = ['BinaryLight-5022136575', 'binaryLight.powerStatus', 'true']
    # ret = home.intervention(evidence)
    # print(ret)

    # TEST do, status: working
    # evidence = {'O': 300.0}
    # ret = home.do(evidence)
    # print(ret)

    # TEST simulate, status: working
    # evidence = {'H': 1}
    # df = home.simulate(evidence, 10)
    # print(df)

    # TEST intervention
    evidence = {'H': 0}
    # print(home.do(evidence=evidence, do_size=10,  resp_time=3))
    home.intervention(evidence)
    print(home.sample())

