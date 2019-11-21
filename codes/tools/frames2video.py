import os, sys
import utils.crash_on_ipy

def get_crf():
    video_info_path = '/workspace/nas_mengdongwei/mmsr/results/ai4khdr_test_191119-161224/videos/video_info'
    video_info = open(video_info_path, 'r').readlines()
    crfs = []
    for ii, info in enumerate(video_info):
        info = info.strip()
        crf = int(info.split()[-1])
        crfs.append(crf)
        print(ii, crf)
    return crfs


def main(in_dir, out_dir):
    file_list = os.listdir(in_dir)
    file_list.sort()
    crfs = get_crf()
    ii = 0
    for _, sub_dir in enumerate(file_list):
        frame_dir = os.path.join(in_dir, sub_dir)
        if os.path.isfile(frame_dir):
            print(frame_dir)
            continue
        if sub_dir == 'videos':
            continue

        video_prefix = os.path.join(out_dir, sub_dir)
        crf = crfs[ii]
        ii += 1
        if sub_dir != '61883118' and sub_dir != '66042767':
            continue

        sh_conmand = 'ffmpeg -r 24000/1001 -i {}/{}_%6d.png -vcodec libx265 -pix_fmt yuv422p -crf {} {}.mp4'.format(frame_dir, sub_dir, crf, video_prefix)
        print(sh_conmand)
        #if crf == 5:
        #    continue
        os.system(sh_conmand)

def test_one(path, out_dir):
    sub_dir = os.path.basename(path)
    prefix_name = sub_dir.split('.mkv')[0]
    save_img_dir = os.path.join(out_dir, prefix_name)
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)

    sh_conmand = 'ffmpeg -i %s -vf fps=1/5 %s/%%06d.png' % (path, save_img_dir)
    print(sh_conmand)
    os.system(sh_conmand)

if __name__ == '__main__':
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    main(in_dir, out_dir)
    #test_one(in_dir, out_dir)
