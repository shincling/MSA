from pyannote.database import FileFinder
from pyannote.database import get_unique_identifier
from pyannote.database import get_protocol

protocol_name='AMI.SpeakerDiarization.MixHeadset'
protocol = get_protocol(protocol_name, progress=False)

def output(train_or_test):
    print('Begin to output ',train_or_test)
    if train_or_test=='train':
        gen=protocol.train()
    elif train_or_test=='development':
        gen=protocol.development()
    elif train_or_test=='test':
        gen=protocol.test()
    idx=0
    while True:
        try:
            cc=next(gen) # including an ordered Dict with annotations we need.
            print(cc['uri'])
            idx+=1

        except StopIteration:
            break
    print(idx)

output('train')
output('development')
output('test')
