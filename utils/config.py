# PARAMETERS
parameters = {
    'max_cond_vars': 4,
    'do_size': 10,
    'do_conf': 0.9,
    'ci_conf': 0.2
}

resp_time = 0

# DEVICES
read_kind = {"T": 'device',
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

read_devices = {"Thermometer-b1f74267ed": 'thermometer.currentTemperature',
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

read_devices_short = {
    "S": "ToggleSwitch-d9704c5a0f",
    "W": "DoorWindowSensor-b041e06747",
    "CO": "COGasSensor-1bc0572e9e",
    "B": "PushButton-5ec148b252",
    "CO2": "CO2GasSensor-b66d761212",
    "L": "BinaryLight-5022136575",
    "H": "Heater-1bd6fdc99a",
    "C": "Cooler-ce2209b064",
    "O": "Thermometer-b1f74267ed",
    "A": "Siren-d6c226252a",
    "Pr": "PresenceSensor-cd52a5fed8",
    "T": "Thermometer-829cc07927"
}

# SOCKET
HOST = "127.0.0.1"
PORT = 7777
