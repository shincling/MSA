#coding=utf8
import os
import subprocess
import shutil
from tqdm import tqdm
from pyannote.database import get_protocol

protocol_name='AMI.SpeakerDiarization.MixHeadset'
protocol = get_protocol(protocol_name, progress=False)

def output(meeting_uri):
    for gen in [protocol.train(),protocol.development(),protocol.test()]:
        while True:
            try:
                cc=next(gen) # including an ordered Dict with annotations we need.
                if meeting_uri==cc['uri'].split('.')[0]:
                    return cc
            except StopIteration:
                break

def cut_with_annotations(aimed_uri,path_to_uris):
    uri=aimed_uri['uri'].split('.')[0]
    annotation=aimed_uri['annotation']._tracks
    times=[i for i in annotation]
    names=[list(i.values())[0] for i in annotation.values()]
    assert len(times)==len(names)
    print(uri,': times and names got with length of ',len(names))
    file_name=path_to_uris+uri+'/video/{}.Overview1.avi'.format(uri)
    file_name_audio=path_to_uris+uri+'/audio/{}.Mix-Headset.wav'.format(uri)
    output_path='./aim_sets/'+uri+'/cutted/'

    if os.path.exists(output_path):
        print(" cleanup: " + output_path)
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    # Begin to cut.
    assert uri in os.listdir(path_to_uris)
    idx=1
    # TODO: Don't do the video cutter, just extract the images directly.
    for time,name in tqdm(zip(times,names)):
        start_time,end_time=time[0],time[1]o
        with open('cutting_log', "w") as ffmpeg_log:
            video_cut_command = ['ffmpeg',
                                 '-ss',  str(start_time),
                                 '-i', file_name,  # input file
                                 '-ss', '0',
                                 '-t', str(end_time-start_time),
                                 # '-c:v libx264 -c:a ac -strict experimental',
                                 '-an',
                                 '-y', output_path+str(idx)+'_'+name+'_'\
                                       +str(round(start_time,3))+'_'+str(round(end_time,3))+'.avi'
                                 ]
            subprocess.call(video_cut_command, stdout=ffmpeg_log, stderr=ffmpeg_log)

        with open('cutting_log', "w") as ffmpeg_log:
            audio_cut_command = ['ffmpeg',
                                 '-ss',  str(start_time),
                                 '-i', file_name_audio,  # input file
                                 '-ss', '0',
                                 '-t', str(end_time-start_time),
                                 # '-c:v libx264 -c:a ac -strict experimental',
                                 '-vn',
                                 '-y', output_path+str(idx)+'_'+name+'_' \
                                 +str(round(start_time,3))+'_'+str(round(end_time,3))+'.wav'
                                 ]
            subprocess.call(audio_cut_command, stdout=ffmpeg_log, stderr=ffmpeg_log)
            idx+=1

# print(output('TS3003a'))

path_to_uris='/home/user/shijing/datasets/AMI/amicorpus/'

aimed_uri=output('TS3003a')
cut_with_annotations(aimed_uri,path_to_uris)
