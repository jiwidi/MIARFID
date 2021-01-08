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
    
    async def on_start(self):
        await super().on_start()
        # Lista de tuplas donde se guardará el id y la posicion de los transportes
        self.set("posiciones", [])

    async def run(self):
        if not self.agent.registration:
            await self.send_registration()

        msg = await self.receive(timeout=5)
        logger.debug("Manager received message: {}".format(msg))

        if msg:
            performative = msg.get_metadata("performative")
            posiciones = self.get("posiciones")
            content = json.loads(msg.body)

            if performative == PROPOSE_PERFORMATIVE:
                # Si la performativa es de propuesta es que hemos recibido un mensaje de un transporte
                # (ver modificacion de la estrategia de transporte)

                # Hay que añadir la posicion del transporte en caso de no tenerla ya
                pos_transporte = content.get("position")
                if (str(msg.sender) not in [x[0] for x in posiciones]):
                    posiciones.append((str(msg.sender), pos_transporte))
                # Actualizamos la lista
                self.set("posiciones", posiciones) 

            else:
                if len(posiciones) > 0:
                    # El mensaje lo ha enviado un cliente
                    # Calculamos la distancia desde la posicion de la persona a los transportes y
                    # escogemos el transporte más cercano (probar tambien + de 1)
                    pos_customer = content["origin"]

                    # Lista para almacenar las distancias calculadas y luego escoger al transporte más cercano
                    distancias = []

                    for t in posiciones:
                        dist = distance_in_meters(t[1], pos_customer)
                        distancias.append((t[0], dist, t[1])) # Guardamos id, distancia y posicion del transporte
                    
                    distancias.sort(key = lambda x: x[1]) # Ordenamos por distancia
                    
                    cercano = distancias.pop(0) # Sacamos al transporte más cercano
                    posiciones.remove((cercano[0], cercano[2])) # Eliminamos la posición de la lista 

                    self.set("posiciones", posiciones) # Actualizamos la lista

                    msg.to = cercano[0]
                    logger.debug("Manager sent request to transport {}".format(cercano[0]))
                    await self.send(msg)
               
################################################################
#                                                              #
#                         Transport Strategy                   #
#                                                              #
################################################################

class MyTransportStrategy(TransportStrategyBehaviour):

    async def run(self):
        # Hay que actualizar la posicion del manager cuando el transporte está libre (cuando no se está moviendo)
        if self.agent.status == TRANSPORT_WAITING:
            await self.send_proposal(self.agent.fleetmanager_id, { "position": self.agent.get_position() })

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
                        await self.send_proposal(content["customer_id"],{})
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
                    await self.accept_transport(transport_id)
                    self.agent.status = CUSTOMER_ASSIGNED
                else:
                    await self.refuse_transport(transport_id)

            elif performative == CANCEL_PERFORMATIVE:
                if self.agent.transport_assigned == str(transport_id):
                    logger.warning(
                        "Customer {} received a CANCEL from Transport {}.".format(self.agent.name, transport_id))
                    self.agent.status = CUSTOMER_WAITING