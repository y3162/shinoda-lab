import json
from src.mp_senet.model.model import MPNet, pesq_score, phase_losses
from src.mp_senet.model.discriminator import MetricDiscriminator, batch_pesq
from src.mp_senet.utils import AttrDict


def main():
    with open('./src/mp_senet/model/config.json', 'r') as f:
        config_json = json.load(f)
    h = AttrDict(config_json)

    generator = MPNet(h)
    discriminator = MetricDiscriminator()

    print(generator)
    print(discriminator)


if __name__ == '__main__':
    main()
