#coding=utf8
import os
import subprocess
import shutil
from tqdm import tqdm
from pyannote.database import get_protocol
import time
from multiprocessing import Process
import threading
import time

protocol_name='AMI.SpeakerDiarization.MixHeadset'
protocol = get_protocol(protocol_name, progress=False)

def output(meeting_uri):
    # To extract the annotations for the given meeting uri.
    for gen in [protocol.train(),protocol.development(),protocol.test()]:
        while True:
            try:
                cc=next(gen) # including an ordered Dict with annotations we need.
                if meeting_uri==cc['uri'].split('.')[0]:
                    return cc
            except StopIteration:
                break

def cut_with_annotations(aimed_uri,path_to_uris,video_label):
    uri=aimed_uri['uri'].split('.')[0]
    annotation=aimed_uri['annotation']._tracks
    times=[i for i in annotation]
    names=[list(i.values())[0] for i in annotation.values()]
    assert len(times)==len(names)
    print(uri,': times and names got with length of ',len(names))
    file_name=path_to_uris+uri+'/video/{}.{}.avi'.format(uri,video_label)
    file_name_audio=path_to_uris[:-1]+'_audio/'+uri+'/audio/{}.Mix-Headset.wav'.format(uri)
    output_path='./aim_sets/'+uri+'/cutted_{}/'.format(video_label)

    if os.path.exists(output_path):
        print(" cleanup: " + output_path)
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    # Begin to cut.
    assert uri in os.listdir(path_to_uris)
    idx=1
    # TODO: Don't do the video cutter, just extract the images directly.
    end_time_pre=0
    output_file_name_pre=''
    counter=0
    for time,name in tqdm(zip(times,names)):
        start_time,end_time=time[0],time[1]
        if end_time-start_time>1: #calculate the ratio of length larger than 1s. About 60%.
            counter+=1
        if start_time<end_time_pre: # if overlapped
            # print('The {} with start time {} Overlaped.'.format(idx,start_time))
            if os.path.exists(output_file_name_pre+'.avi'):
                # print('Remove previous file.',output_file_name_pre)
                os.remove(output_file_name_pre+'.avi')
                os.remove(output_file_name_pre+'.wav')
                shutil.rmtree(output_file_name_pre)
            end_time_pre=end_time
            idx+=1
            continue
        with open('cutting_log', "w") as ffmpeg_log:
            output_file_name_pre=output_path+str(idx)+'_'+name+'_'\
                                 +str(round(start_time,3))+'_'+str(round(end_time,3))
            video_cut_command = ['ffmpeg',
                                 '-ss',  str(start_time),
                                 '-i', file_name,  # input file
                                 '-ss', '0',
                                 '-t', str(end_time-start_time),
                                 # '-c:v libx264 -c:a ac -strict experimental',
                                 '-an',
                                 '-y', output_path+str(idx)+'_'+name+'_'\
                                       +str(round(start_time,3))+'_'+str(round(end_time,3))+'.avi'
                                       # +str(round(start_time,3))+'_'+str(round(end_time,3))+'/%06d.jpeg'
                                 ]
            subprocess.call(video_cut_command, stdout=ffmpeg_log, stderr=ffmpeg_log)

        with open('cutting_log', "w") as ffmpeg_log:
            os.mkdir(output_path+str(idx)+'_'+name+'_' \
                                 +str(round(start_time,3))+'_'+str(round(end_time,3)))
            image_cut_command = ['ffmpeg',
                                 '-ss',  str(start_time),
                                 '-t', str(end_time-start_time),
                                 '-i', file_name,  # input file
                                 # '-c:v libx264 -c:a ac -strict experimental',
                                 '-an',
                                 '-y', output_path+str(idx)+'_'+name+'_' \
                                 +str(round(start_time,3))+'_'+str(round(end_time,3))+'/'\
                                 +str(idx)+'_'+name+'_' \
                                 +str(round(start_time,3))+'_'+str(round(end_time,3))+'_%06d.jpeg'
                                 ]
            # print(image_cut_command)
            subprocess.call(image_cut_command, stdout=ffmpeg_log, stderr=ffmpeg_log)

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
        end_time_pre=end_time
    print('Ratio of the counter.',float(counter/idx))

# print(output('TS3003a'))

path_to_uris='/home/user/shijing/datasets/AMI/amicorpus/'
''' Dir structure:
root
  --amicorpus
    --urisXXXX/video
  --amicorpus_audio
    --urisXXXX/audio
'''
uri_list=os.listdir(path_to_uris)
btime=time.time()
def one_step(uri):
    print('*'*40)
    print('Begin to conduct uri:',uri)
    try:
        aimed_uri=output(uri)
        if uri[0]=='E':
            video_label='Corner'
            cut_with_annotations(aimed_uri,path_to_uris,video_label)
        elif uri[0]=='I':
            video_label='C'
            cut_with_annotations(aimed_uri,path_to_uris,video_label)
        elif uri[0]=='T':
            video_label='Overview1'
            cut_with_annotations(aimed_uri,path_to_uris,video_label)
            video_label='Overview2'
            cut_with_annotations(aimed_uri,path_to_uris,video_label)
        else:
            print('Wrong uri name.')
            1/0
    except Exception as result:
        print('Erros occor here~!!!!!')
        print(result)
    print('Takes total time:',time.time()-btime)

for uri in uri_list:
    # one_step(uri)
    p=Process(target=one_step,args=(uri,))
    p.start()

print('Takes total time:',time.time()-btime)
# aimed_uri=output('TS3003a')
# cut_with_annotations(aimed_uri,path_to_uris)
