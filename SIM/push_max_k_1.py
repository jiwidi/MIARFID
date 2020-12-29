import datetime
import json
import random
import time
import click

import spade


class PushAgent(spade.agent.Agent):

    async def setup(self):
        self.value = random.randint(1, 1000)

        start_at = datetime.datetime.now() + datetime.timedelta(seconds=5)
        self.add_behaviour(self.PushBehaviour(period=2, start_at=start_at))
        template = spade.template.Template(metadata={"performative": "PUSH"})
        self.add_behaviour(self.RecvBehaviour(), template)

        print("{} ready.".format(self.name))

    def add_value(self, value):
        self.value = max(self.value, value)

    def add_contacts(self, contact_list):
        self.contacts = [c.jid for c in contact_list if c.jid != self.jid]
        self.length = len(self.contacts)

    class PushBehaviour(spade.behaviour.PeriodicBehaviour):

        async def run(self):
            #k = random.randint(1, self.agent.length)
            k=1
            print("{} period with k={}!".format(self.agent.name, k))
            random_contacts = random.sample(self.agent.contacts, k)
            print("{} sending to {}".format(self.agent.name, [x.localpart for x in random_contacts]))

            for jid in random_contacts:
                body = json.dumps({"value": self.agent.value, "timestamp": time.time()})
                msg = spade.message.Message(to=str(jid), body=body, metadata={"performative": "PUSH"})
                await self.send(msg)

    class RecvBehaviour(spade.behaviour.CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=2)
            if msg:
                body = json.loads(msg.body)
                self.agent.add_value(body["value"])
                #print("[{}] <{}>".format(self.agent.name, self.agent.value))


@click.command()
@click.option('--count', default=10, help='Number of agents.')
def main(count):
    agents = []
    print("Creating {} agents...".format(count))
    for x in range(1, count + 1):
        print("Creating agent {}...".format(x))
        agents.append(PushAgent("push_agent_1626_{}@localhost".format(x), "test"))

    for ag in agents:
        ag.add_contacts(agents)
        ag.value = 0

    for ag in agents:
        ag.start()
    time.sleep(5)
    while True:
        try:
            time.sleep(1)
            status = [ag.value for ag in agents]
            print("STATUS: {}".format(status))
            if len(set(status)) <= 1:
                print("Gossip done.")
                break
        except KeyboardInterrupt:
            break

    for ag in agents:
        ag.stop()
    print("Agents finished")


if __name__ == '__main__':
    main()