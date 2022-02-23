from requests import put, post, get
import json

# Reference Link
# http://adeleresearchgroup.github.io/iCasa/snapshot/rest-api.html

root = f'http://localhost:9000'
get_devices = f'{root}/icasa/devices/devices'
get_device = f'{root}/icasa/devices/device'
get_zones = f'{root}/icasa/zones/zones'
get_zone = f'{root}/icasa/zones/zone'
get_persons = f'{root}/icasa/persons/persons'
get_person = f'{root}/icasa/persons/person'
put_device = f'{root}/icasa/devices/device'
put_zone = f'{root}/icasa/zones/zone'
put_person = f'{root}/icasa/persons/person'


# Test PUT
# person -> ok
# other puts have no effects

body_1 = {'heater.powerLevel': 300.0}
body_2 = {'id': 'heater', 'heater.powerLevel': 300.0}
body_3 = {'id': 'heater', 'properties': {'name': 'heater.powerLevel', 'value': 300.0}}

binLight = {
      "positionY":199,
      "name":"BinaryLight-5022136xxx",
      "fault":"no",
      "location":"room",
      "id":"BinaryLight-5022136xxx",
      "state":"activated",
      "services":[
         "fr.liglab.adele.icasa.device.light.BinaryLight",
         "fr.liglab.adele.icasa.device.PowerObservable",
         "fr.liglab.adele.icasa.simulator.SimulatedDevice",
         "fr.liglab.adele.icasa.device.GenericDevice"
      ],
      "type":"iCasa.BinaryLight",
      "properties":[
         {
            "unit":"N/A",
            "name":"binaryLight.maxPowerLevel",
            "value":100
         },
         {
            "unit":"N/A",
            "name":"binaryLight.powerStatus",
            "value":True
         },
         {
            "unit":"N/A",
            "name":"fault",
            "value":"no"
         },
         {
            "unit":"N/A",
            "name":"powerObservable.currentConsumption",
            "value":0
         },
         {
            "unit":"N/A",
            "name":"state",
            "value":"activated"
         },
         {
            "unit":"N/A",
            "name":"Location",
            "value":"room"
         }
      ],
      "positionX":426
   }

heater = {
      "positionY":238,
      "name":"Heater-1bd6fdc99a",
      "fault":"no",
      "location":"room",
      "id":"Heater-1bd6fdc99a",
      "state":"activated",
      "services":[
         "fr.liglab.adele.icasa.device.PowerObservable",
         "fr.liglab.adele.icasa.simulator.SimulatedDevice",
         "fr.liglab.adele.icasa.device.temperature.Heater",
         "fr.liglab.adele.icasa.device.GenericDevice"
      ],
      "type":"iCasa.Heater",
      "properties":[
         {
            "unit":"N/A",
            "name":"fault",
            "value":"no"
         },
         {
            "unit":"N/A",
            "name":"heater.maxPowerLevel",
            "value":1000
         },
         {
            "unit":"N/A",
            "name":"powerObservable.currentConsumption",
            "value":0
         },
         {
            "unit":"N/A",
            "name":"state",
            "value":"activated"
         },
         {
            "unit":"N/A",
            "name":"heater.powerLevel",
            "value":555
         },
         {
            "unit":"N/A",
            "name":"Location",
            "value":"room"
         }
      ],
      "positionX":541
   }

# Execute put to update
resp = put(f'{put_device}/Heater-1bd6fdc99a', data=json.dumps(heater))
print(resp.status_code)

# Check if updated successfully
req = get(f'{get_device}/Heater-1bd6fdc99a')
print(req.json())
