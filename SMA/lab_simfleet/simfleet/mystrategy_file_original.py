import json
import random
 
from loguru import logger
 
from simfleet.fleetmanager import FleetManagerStrategyBehaviour
from simfleet.customer import CustomerStrategyBehaviour
from simfleet.helpers import PathRequestException, distance_in_meters, are_close
from simfleet.protocol import REQUEST_PERFORMATIVE, ACCEPT_PERFORMATIVE, REFUSE_PERFORMATIVE, PROPOSE_PERFORMATIVE, \
    CANCEL_PERFORMATIVE, INFORM_PERFORMATIVE, QUERY_PROTOCOL, REQUEST_PROTOCOL
from simfleet.transport import TransportStrategyBehaviour
from simfleet.utils import TRANSPORT_WAITING, TRANSPORT_WAITING_FOR_APPROVAL, CUSTOMER_WAITING, TRANSPORT_MOVING_TO_CUSTOMER, \
    CUSTOMER_ASSIGNED, TRANSPORT_WAITING_FOR_STATION_APPROVAL, TRANSPORT_MOVING_TO_STATION, \
    TRANSPORT_CHARGING, TRANSPORT_CHARGED, TRANSPORT_NEEDS_CHARGING
import math
from spade.message import MessageBase, Message
 
################################################################
#                                                              #
#                     FleetManager Strategy                    #
#                                                              #
################################################################
 
class MyFleetManagerStrategy(FleetManagerStrategyBehaviour):
    async def on_start():
        await super().on_start()
        # Diccionario para almacenar las posiciones de los transportes
        # Cada vez que un transporte envia un mensaje se actualiza el diccionario con su posicion
        self.set("posiciones", {})

    async def run(self):
        if not self.agent.registration:
            await self.send_registration()

        msg = await self.receive(timeout=5)
        logger.debug("Manager received message: {}".format(msg))

        if msg:
            content = json.loads(msg.body)
            transportes = self.get_transport_agents().values() # Objetos TransportAgent
            ids_transportes = [t["jid"] for t in transportes] # Lista con los id
            
            if str(msg.sender) in ids_transportes: 
                # Si he recibido mensaje de un transporte
                # Almacenar posicion en el diccionario
                posiciones = self.get("posiciones")  # Diccionario con las posiciones
                pos_transporte = content["position"] # Posicion actual del transporte
                posiciones[str(msg.sender)] = pos_transporte
                self.set("posiciones", posiciones)
            else:
                # He recibido un mensaje de un cliente
                # Calculo las distancias a los transportes y se propaga el mensaje a todos los transportes de manera ordenada
                posiciones = self.get("posiciones")
                pos_customer = content["origin"]
                # Almacenaremos en una lista de tuplas los id de los transportes y la distancia a ellos
                distancias = []
                for id, pos in posiciones.items():
                    dist = distance_in_meters(pos, pos_customer)
                    distancias.append(id, dist)
                # Ordenamos la lista de distancias
                distancias = sorted(distancias, key=lambda x:x[1])
                # Enviamos mensaje a todos los transportes en orden de cercania
                for t in distancias:
                    msg.to = str(t[0])
                    logger.debug("Manager sent request to transport {}".format(transport["name"]))
                    await self.send(msg)
                    


        #if msg:
        #    for transport in self.get_transport_agents().values():
        #        msg.to = str(transport["jid"])
        #        logger.debug("Manager sent request to transport {}".format(transport["name"]))
        #        await self.send(msg)
               
################################################################
#                                                              #
#                         Transport Strategy                   #
#                                                              #
################################################################

class MyTransportStrategy(TransportStrategyBehaviour):
    async def on_start():
        await super().on_start()
        # Al iniciar hay que enviar un mensaje con la posicion del agente al Manager
        content = {
            "name": self.agent.name,
            "jid": str(self.agent.jid),
            "fleet_type": self.agent.fleet_type,
            "position": self.agent.get_position()
        }
        msg = Message()
        msg.to = str(self.agent.fleetmanager_id)
        msg.set_metadata("protocol", REQUEST_PROTOCOL)
        msg.set_metadata("performative", INFORM_PERFORMATIVE)
        msg.body = json.dumps(content)
        await self.send(msg)

    async def run(self):
        # Hay que actualizar la posicion del manager cuando el transporte está libre (cuando no se está moviendo)
        # Enviamos un mensaje igual que en on_start()
        if (self.agent.status != TRANSPORT_MOVING_TO_CUSTOMER) and (self.agent.status != TRANSPORT_MOVING_TO_STATION):
            content = {
                "name": self.agent.name,
                "jid": str(self.agent.jid),
                "fleet_type": self.agent.fleet_type,
                "position": self.agent.get_position()
            }
            msg = Message()
            msg.to = str(self.agent.fleetmanager_id)
            msg.set_metadata("protocol", REQUEST_PROTOCOL)
            msg.set_metadata("performative", INFORM_PERFORMATIVE)
            msg.body = json.dumps(content)
            await self.send(msg)

        if self.agent.needs_charging():
            if self.agent.stations is None or len(self.agent.stations) < 1:
                logger.warning("Transport {} looking for a station.".format(self.agent.name))
                await self.send_get_stations()
            else:
                station = random.choice(list(self.agent.stations.keys()))
                logger.info("Transport {} reserving station {}.".format(self.agent.name, station))
                await self.send_proposal(station)
                self.agent.status = TRANSPORT_WAITING_FOR_STATION_APPROVAL

        msg = await self.receive(timeout=5)
        if not msg:
            return
        logger.debug("Transport received message: {}".format(msg))
        try:
            content = json.loads(msg.body)
        except TypeError:
            content = {}

        performative = msg.get_metadata("performative")
        protocol = msg.get_metadata("protocol")

        if protocol == QUERY_PROTOCOL:
            if performative == INFORM_PERFORMATIVE:
                self.agent.stations = content
                logger.info("Got list of current stations: {}".format(list(self.agent.stations.keys())))
            elif performative == CANCEL_PERFORMATIVE:
                logger.info("Cancellation of request for stations information.")

        elif protocol == REQUEST_PROTOCOL:
            logger.debug("Transport {} received request protocol from customer/station.".format(self.agent.name))

            if performative == REQUEST_PERFORMATIVE:
                if self.agent.status == TRANSPORT_WAITING:
                    if not self.has_enough_autonomy(content["origin"], content["dest"]):
                        await self.cancel_proposal(content["customer_id"])
                        self.agent.status = TRANSPORT_NEEDS_CHARGING
                    else:
                        await self.send_proposal(content["customer_id"])
                        self.agent.status = TRANSPORT_WAITING_FOR_APPROVAL

            elif performative == ACCEPT_PERFORMATIVE:
                if self.agent.status == TRANSPORT_WAITING_FOR_APPROVAL:
                    logger.debug("Transport {} got accept from {}".format(self.agent.name,
                                                                          content["customer_id"]))
                    try:
                        self.agent.status = TRANSPORT_MOVING_TO_CUSTOMER
                        await self.pick_up_customer(content["customer_id"], content["origin"], content["dest"])
                    except PathRequestException:
                        logger.error("Transport {} could not get a path to customer {}. Cancelling..."
                                     .format(self.agent.name, content["customer_id"]))
                        self.agent.status = TRANSPORT_WAITING
                        await self.cancel_proposal(content["customer_id"])
                    except Exception as e:
                        logger.error("Unexpected error in transport {}: {}".format(self.agent.name, e))
                        await self.cancel_proposal(content["customer_id"])
                        self.agent.status = TRANSPORT_WAITING
                else:
                    await self.cancel_proposal(content["customer_id"])

            elif performative == REFUSE_PERFORMATIVE:
                logger.debug("Transport {} got refusal from customer/station".format(self.agent.name))
                self.agent.status = TRANSPORT_WAITING

            elif performative == INFORM_PERFORMATIVE:
                if self.agent.status == TRANSPORT_WAITING_FOR_STATION_APPROVAL:
                    logger.info("Transport {} got accept from station {}".format(self.agent.name,
                                                                                 content["station_id"]))
                    try:
                        self.agent.status = TRANSPORT_MOVING_TO_STATION
                        await self.send_confirmation_travel(content["station_id"])
                        await self.go_to_the_station(content["station_id"], content["dest"])
                    except PathRequestException:
                        logger.error("Transport {} could not get a path to station {}. Cancelling..."
                                     .format(self.agent.name, content["station_id"]))
                        self.agent.status = TRANSPORT_WAITING
                        await self.cancel_proposal(content["station_id"])
                    except Exception as e:
                        logger.error("Unexpected error in transport {}: {}".format(self.agent.name, e))
                        await self.cancel_proposal(content["station_id"])
                        self.agent.status = TRANSPORT_WAITING
                elif self.agent.status == TRANSPORT_CHARGING:
                    if content["status"] == TRANSPORT_CHARGED:
                        self.agent.transport_charged()
                        await self.agent.drop_station()

            elif performative == CANCEL_PERFORMATIVE:
                logger.info("Cancellation of request for {} information".format(self.agent.fleet_type))    
 
################################################################
#                                                              #
#                       Customer Strategy                      #
#                                                              #
################################################################

class MyCustomerStrategy(CustomerStrategyBehaviour):

    async def run(self):
        if self.agent.fleetmanagers is None:
            await self.send_get_managers(self.agent.fleet_type)

            msg = await self.receive(timeout=5)
            if msg:
                performative = msg.get_metadata("performative")
                if performative == INFORM_PERFORMATIVE:
                    self.agent.fleetmanagers = json.loads(msg.body)
                    return
                elif performative == CANCEL_PERFORMATIVE:
                    logger.info("Cancellation of request for {} information".format(self.agent.type_service))
                    return

        if self.agent.status == CUSTOMER_WAITING:
            await self.send_request(content={})

        msg = await self.receive(timeout=5)

        if msg:
            performative = msg.get_metadata("performative")
            transport_id = msg.sender
            if performative == PROPOSE_PERFORMATIVE:
                if self.agent.status == CUSTOMER_WAITING:
                    # He recibido una propuesta de algun transporte
                    logger.debug(
                        "Customer {} received proposal from transport {}".format(self.agent.name, transport_id))
                    await self.accept_transport(transport_id) # Aceptamos la propuesta al más cercano
                    self.agent.status = CUSTOMER_ASSIGNED
                else:
                    await self.refuse_transport(transport_id)

            elif performative == CANCEL_PERFORMATIVE:
                if self.agent.transport_assigned == str(transport_id):
                    logger.warning(
                        "Customer {} received a CANCEL from Transport {}.".format(self.agent.name, transport_id))
                    self.agent.status = CUSTOMER_WAITING