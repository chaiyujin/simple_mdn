import os
import cv2
import dde
import ffmpy
import numpy
import fbxanime
import scipy.io.wavfile as wav
from utils import console
from python_speech_features import mfcc, delta, logfbank

loglevel = 'error'


def remove_files(list):
    for file_name in list:
        if os.path.exists(file_name):
            os.remove(file_name)


def demux_video(video_path, clear_old=False):
    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        return None
    file_prefix, _ = os.path.splitext(video_path)
    print('\033[01;32m[Demuxing]\033[0m ' + video_path)
    v_path = file_prefix + '_video.mp4'
    a_path = file_prefix + '_audio.mp4'
    w_path = file_prefix + '_audio.wav'
    if clear_old:
        remove_files([v_path, a_path, w_path])
    # 1. demux audio
    if not os.path.exists(w_path):
        ffmpy.FFmpeg(
            inputs={video_path: None},
            outputs={
                a_path: '-map 0:1 -c:a copy -f mp4 -loglevel ' + loglevel}
        ).run()
        # convert audio into wav
        ffmpy.FFmpeg(
            inputs={a_path: None},
            outputs={
                w_path: '-acodec pcm_s16le -ac 1 -ar 16000 -loglevel ' +
                loglevel}
        ).run()
        # delete middle result
        os.remove(a_path)
    # 2. resample video
    cap = cv2.VideoCapture(video_path)
    if cap.get(cv2.CAP_PROP_FPS) != 25.0:
        if not os.path.exists(v_path):
            ffmpy.FFmpeg(
                inputs={video_path: None},
                outputs={v_path: '-qscale 0 -r 25 -y -loglevel ' + loglevel}
            ).run()
    else:
        v_path = video_path
    cap.release()

    return {'video_path': v_path, 'audio_path': w_path}


def get_mfcc(audio_path):
    fs, audio = wav.read(audio_path)
    assert(fs == 16000)
    mfcc_feat = mfcc(audio, samplerate=fs, winlen=0.04, winstep=0.04)
    # d_mfcc_feat = delta(mfcc_feat, 2)
    # fbank_feat = logfbank(audio, samplerate=fs)
    return mfcc_feat


def cache_path(video_path):
    return video_path + '_cache_expr'


def get_anime_data_from_cache(video_path, clear_old):
    anime_data = None
    if clear_old:
        return anime_data
    cache = cache_path(video_path)
    if os.path.exists(cache):
        console.log('log', 'Get anime data from cache\n')
        anime_data = []
        with open(cache) as file:
            for line in file:
                exprs = line.strip().split(' ')
                tuple = []
                for expr in exprs:
                    tuple.append(float(expr))
                assert(len(tuple) == 19)
                tuple = numpy.asarray(tuple, dtype=numpy.float32)
                anime_data.append(tuple)
    return anime_data


def save_anime_data_into_cache(video_path, anime_data):
    cache = cache_path(video_path)
    with open(cache, 'w') as file:
        for data in anime_data:
            for d in data:
                file.write(str(d) + ' ')
            file.write('\n')


def get_anime_data_from_video(video_path, show, fbx):
    console.log('log', 'Get anime data from video\n')
    cap = cv2.VideoCapture(video_path)
    anime_data = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ret = False
        for _ in range(200):
            ret |= dde.run(frame)
            if ret:
                break
        if not ret:
            print('\033[01;31m[Bad frame]\033[0m in ' + video_path)
            return None
        for _ in range(8):
            dde.run(frame)

        # push anime data
        expr = dde.get('expression_raw')
        anime_data.append(numpy.asarray(
            dde.expr_to_mouth(expr),
            dtype=numpy.float32))

        if show:
            lm = dde.get('landmarks')
            for i in range(0, lm.shape[0], 2):
                x = lm[i]
                y = lm[i + 1]
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.imshow('frame', frame)

        if fbx:
            mat = fbxanime.get_image(expr)
            cv2.imshow('fbx', mat)

        if show or fbx:
            cv2.waitKey(1)

    cap.release()
    return anime_data


def process_media(video_path, show=False, fbx=False, clear_old=False):
    dde.reset()
    demuxed = demux_video(video_path, clear_old)
    # audio feat
    audio_feat = get_mfcc(demuxed['audio_path'])
    a_frames = len(audio_feat)
    if a_frames <= 0:
        print('\033[01;31m[No audio feature]\033[0m ' +
              video_path)
        return None
    # video track
    anime_data = get_anime_data_from_cache(
        demuxed['video_path'], clear_old
    )
    if anime_data is None:
        anime_data = get_anime_data_from_video(
            demuxed['video_path'], show, fbx
        )
    if anime_data is None:
        return None
    v_frames = len(anime_data)
    if v_frames != a_frames:
        print('\033[01;31m[Fail to align]\033[0m ' +
              video_path)
        return None
    save_anime_data_into_cache(
        demuxed['video_path'],
        anime_data
    )
    anime_data = numpy.asarray(anime_data, dtype=numpy.float32)
    return (audio_feat, anime_data)


if __name__ == '__main__':
    dde.init('D:/software/dev/DDE/v3.bin')
    dde.set_n_copies(100)
    fbxanime.init(1280, 720, 'fbx_anime.fbx')
    process_media('bbaf2n.mpg')
