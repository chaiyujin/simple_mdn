from __future__ import absolute_import, division

import os
import cv2
import dde
import ffmpy
import numpy
import fbxanime
import scipy.io.wavfile as wav
from . import console
from python_speech_features import mfcc, delta, logfbank

VIDEO_FPS = 25.0
loglevel = 'error'
g_inited = False
g_fbx_w = 0
g_fbx_h = 0


def init_dde_fbx(
        dde_path='D:/software/dev/DDE/v3.bin',
        fbx_path='D:/todo/DeepLearning/projects/End/asset/fbx_anime.fbx',
        fbx_w=1280, fbx_h=720):
    global g_inited
    global g_fbx_w
    global g_fbx_h

    if g_inited:
        return True
    if os.path.exists(dde_path) and os.path.exists(fbx_path):
        dde.init(dde_path)
        dde.set_n_copies(120)
        fbxanime.init(fbx_w, fbx_h, fbx_path)
        g_fbx_w = fbx_w
        g_fbx_h = fbx_h
        g_inited = True
        return True
    else:
        console.log('error', 'Missing', 'Cannot find dde or fbx init file.\n')
        return False


def remove_files(list):
    for file_name in list:
        if os.path.exists(file_name):
            os.remove(file_name)


def demux_video(video_path, clear_old=False):
    global VIDEO_FPS

    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        return None
    file_prefix, _ = os.path.splitext(video_path)
    console.log('log', 'Demuxing', video_path + '\n')
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
    if cap.get(cv2.CAP_PROP_FPS) != VIDEO_FPS:
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
    fbank_feat = logfbank(audio, samplerate=fs, winlen=0.04, winstep=0.04)
    ret = numpy.concatenate((mfcc_feat, fbank_feat), 1)
    return ret


def cache_path(video_path):
    return video_path + '_cache_expr'


def get_anime_data_from_cache(video_path, clear_old):
    anime_data = None
    if clear_old:
        return anime_data
    cache = cache_path(video_path)
    if os.path.exists(cache):
        console.log('log', 'Get anime data from cache')
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
    if anime_data is None:
        anime_data = []
    cache = cache_path(video_path)
    with open(cache, 'w') as file:
        for data in anime_data:
            for d in data:
                file.write(str(d) + ' ')
            file.write('\n')


def noop_scan(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for _ in range(200):
            ret |= dde.run(frame)
            if ret:
                break
    cap.release()


def get_anime_data_from_video(video_path, show, fbx, sil_frames):
    # dde.reset()
    noop_scan(video_path)
    console.log('log', 'Get anime data from video')
    cap = cv2.VideoCapture(video_path)
    anime_data = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in sil_frames:
            # print('sil')
            expr = numpy.zeros((46), dtype=numpy.float32)
            anime_data.append(numpy.zeros(
                (19),
                dtype=numpy.float32
            ))
        else:
            ret = False
            for _ in range(10):
                ret |= dde.run(frame)
                if ret:
                    break
            if not ret:
                console.log('error', 'Bad Frame', video_path + '\n')
                return None

            # print('speak')
            for _ in range(4):
                dde.run(frame)

            # push anime data
            expr = dde.get('expression_raw')
            anime_data.append(numpy.asarray(
                dde.expr_to_mouth(expr),
                dtype=numpy.float32))

        frame_id += 1

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


def process_media(video_path, config, show=False, fbx=False, clear_old=False):
    dde.reset()
    demuxed = demux_video(video_path, clear_old)
    # audio feat
    audio_feat = get_mfcc(demuxed['audio_path'])
    a_frames = len(audio_feat)
    if a_frames <= 0:
        console.log('error', 'No audio feature', video_path + '\n')
        return None
    # get align text
    file_prefix, _ = os.path.splitext(video_path)
    align_text = file_prefix + '.align'
    sil_frames = []
    with open(align_text, 'r') as file:
        count = 0
        for line in file:
            line = line.strip().split(' ')
            start = int(float(line[0]) / 1000.0)
            end = int(float(line[1]) / 1000.0)
            word = line[2]
            if word == 'sil':
                if count == 0:
                    end = end
                else:
                    end = end + 1
                count += 1
                for i in range(start, end):
                    sil_frames.append(i)
    # video track
    anime_data = get_anime_data_from_cache(
        demuxed['video_path'], clear_old
    )
    if anime_data is None:
        anime_data = get_anime_data_from_video(
            demuxed['video_path'], show, fbx,
            sil_frames=sil_frames
        )
    # save the expr into cache file to speed up
    save_anime_data_into_cache(
        demuxed['video_path'],
        anime_data
    )
    if anime_data is None:
        return None

    for idx in sil_frames:
        if (idx >= len(anime_data)):
            return None
        anime_data[idx] = numpy.zeros(
            anime_data[idx].shape, dtype=anime_data[idx].dtype)

    v_frames = len(anime_data)
    if v_frames != a_frames:
        console.log('error', 'Fail to align', video_path + '\n')
        return None

    if config['anime_num_features'] < len(anime_data[0]):
        new_data = []
        for i in range(v_frames):
            new_frame = []
            for j in range(config['anime_num_features']):
                new_frame.append(anime_data[i][j])
            new_data.append(new_frame)
        anime_data = new_data

    anime_data = numpy.asarray(anime_data, dtype=numpy.float32)
    is_silent = numpy.zeros((anime_data.shape[0]), dtype=numpy.float32)
    for idx in sil_frames:
        is_silent[idx] = 1
    numpy.clip(anime_data, 0.0, 1.0, out=anime_data)
    print('Audio shape: ', audio_feat.shape)
    print('Anime shape: ', anime_data.shape)
    data_path_prefix, _ = os.path.splitext(demuxed['audio_path'])
    return (audio_feat, anime_data, is_silent, data_path_prefix)


def mux(audio_path, video_path, media_path):
    ffmpy.FFmpeg(
        inputs={audio_path: None, video_path: None},
        outputs={media_path: '-loglevel ' + loglevel}
    ).run()


def generate_video_for_anime(anime_data, video_path):
    global VIDEO_FPS
    global g_fbx_h
    global g_fbx_w

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (g_fbx_w, g_fbx_h))
    for i, frame in enumerate(anime_data):
        frame = numpy.asarray(frame, dtype=numpy.float32)
        frame = dde.mouth_to_expr(frame)
        img = fbxanime.get_image(frame)
        out.write(img)
    out.release()


def sample_video(data, media_path):
    prefix, _ = os.path.splitext(media_path)
    m_path_pred = prefix + '_pred'
    m_path_true = prefix + '_true'
    generate_video_for_anime(data['anime_pred'], m_path_pred + '.avi')
    generate_video_for_anime(data['anime_true'], m_path_true + '.avi')
    # for i in range(len(data['anime_true'])):
    #     d = data['anime_true'][i]
    #     mean = "%.4f" % d.mean()
    #     maxv = "%.4f" % numpy.amax(d)
    #     print(i, ' ', mean, ' ', maxv)
    # print('--------------------')

    if os.path.exists(m_path_true + '.mp4'):
        os.remove(m_path_true + '.mp4')
    if os.path.exists(m_path_pred + '.mp4'):
        os.remove(m_path_pred + '.mp4')
    mux(
        data['path_prefix'] + '.WAV',
        m_path_pred + '.avi',
        m_path_pred + '.mp4'
    )
    mux(
        data['path_prefix'] + '.WAV',
        m_path_true + '.avi',
        m_path_true + '.mp4'
    )
    os.remove(m_path_pred + '.avi')
    os.remove(m_path_true + '.avi')


init_dde_fbx()

if __name__ == '__main__':
    # dde.init('D:/software/dev/DDE/v3.bin')
    # dde.set_n_copies(100)
    # fbxanime.init(1280, 720, 'fbx_anime.fbx')
    # process_media('bbaf2n.mpg')
    anime_data = numpy.zeros(
        (250, 19), dtype=numpy.float32
    )
    generate_video_for_anime(anime_data, 'test.avi')
