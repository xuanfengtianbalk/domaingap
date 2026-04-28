import math

import matplotlib

# matplotlib.use("Agg")
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os
import numpy as np
from rotation_parameter import quaternion2rot, points_trans_accord
from utils_datasets.speedplus_utils_main.utils  import PyTorchSatellitePoseEstimationDataset
import  torch
os.chdir('/opt/dl_workspace/algorithm/04-myself/TestTimeTraining')
print(matplotlib.matplotlib_fname())
plt.rcParams["font.sans-serif"] = "Times New Roman"
# plt.rc('font',**{'family':'serif','serif':['Times']})
points = np.array([[0.38,
                0.385,
                0.325,
                1]
                  ,
               [-0.37,
                0.385,
                0.325,
                1]
                  ,
               [0.38,
                -0.385,
                0.325,
                1]
                  ,
               [-0.37,
                -0.385,
                0.325,
                1]
                  ,
               [0.37,
                0.30,
                0,
                1]
                  ,
               [-0.37,
                0.30,
                0,
                1]
                  ,
               [0.355,
                -0.265,
                0,
                1]
                  ,
               [-0.37,
                -0.30,
                0,
                1]
                  ,
               [0.305,
                -0.57,
                0.255,
                1]
                  ,
               [0.55,
                0.49,
                0.26,
                1]
                  ,
               [-0.53,
                0.48,
                0.26,
                1]]
                  )

def err_cloud_3d(x, y, z, c):
    # ********* Begin *********#

    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection='3d')

    my_cmap = plt.get_cmap('rainbow')
    import math
    # x = [math.log2(abs(t))*t/abs(t) for t in x]
    # y = [math.log2(abs(t))*t/abs(t) for t in y]
    # z = [math.log2(abs(t))*t/abs(t) for t in z]
    sctt = ax.scatter3D(x,y,z, c=c, cmap=my_cmap)

    ax.set_xlabel('X-axis', fontweight='bold')
    ax.set_ylabel('Y-axis', fontweight='bold')
    ax.set_zlabel('Z-axis', fontweight='bold')

    fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)

    # plt.show()
    plt.savefig('T3d.png')

def err_polar_2d(theta_list, phi_list, c):

    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes()
    my_cmap = plt.get_cmap('rainbow')
    sctt = ax.scatter(theta_list,phi_list, c=c, cmap=my_cmap)

    ax.set_xlabel('theta-axis', fontweight='bold')
    ax.set_ylabel('phi-axis', fontweight='bold')
    fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)
    # plt.show()
    plt.savefig('T2d.png')

def err_pixel_2d(pixel_location_x_list, pixel_location_y_list, c):

    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes()
    my_cmap = plt.get_cmap('rainbow')
    sctt = ax.scatter(pixel_location_x_list,pixel_location_y_list, c=c, cmap=my_cmap)

    ax.set_xlabel('x-axis', fontweight='bold')
    ax.set_ylabel('y-axis', fontweight='bold')
    fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)
    # plt.show()
    plt.savefig('Tpixel.png')

def xyz2sphere(xyz):
    x,y,z = xyz
    r = math.sqrt(x*x + y*y + z*z)
    theta = math.acos(z/r)
    phi=math.atan2(y,x)
    return theta,phi

def err_polar_bar(theta_list, phi_list, c):
    #每n度
    theta_tensor = torch.tensor(theta_list)
    c_tensor = torch.tensor(c)
    num=36
    deflaut_clip=math.pi/num
    avg_list = []
    std_list = []
    # i=1
    for i in range(num):
        theta_select = (theta_tensor>(i)*deflaut_clip) & (theta_tensor<(i+1)*deflaut_clip)
        theta_value = c_tensor[theta_select]
        avg = theta_value.mean()
        std = theta_value.std()
        avg_list.append(avg)
        std_list.append(std)


    fig1 = plt.figure(figsize=(12, 9))

    # ax = plt.axes()
    # my_cmap = plt.get_cmap('rainbow')
    plt.errorbar([180*(i+0.5)/num for i in range(num)],avg_list,yerr=std_list,fmt='.k')
    plt.xlabel('theta-axis', fontweight='bold', fontsize=14)
    plt.ylabel('err-axis', fontweight='bold', fontsize=14)
    plt.xticks(np.arange(0,181,10))

    plt.savefig('theta_err_bar.png')
    plt.close(fig1)
    #每30度phi
    phi_tensor = torch.tensor(phi_list)
    c_tensor = torch.tensor(c)
    num=72
    deflaut_clip=2*math.pi/num
    avg_list = []
    std_list = []
    # i=1
    for i in range(num):
        phi_select = (phi_tensor>(i)*deflaut_clip-math.pi) & (phi_tensor<(i+1)*deflaut_clip-math.pi)
        phi_value = c_tensor[phi_select]
        avg = phi_value.mean()
        std = phi_value.std()
        avg_list.append(avg)
        std_list.append(std)


    fig2 = plt.figure(figsize=(12, 9))

    # ax = plt.axes()
    # my_cmap = plt.get_cmap('rainbow')
    plt.errorbar([360*(i+0.5)/num - 180 for i in range(num)],avg_list,yerr=std_list,fmt='.k')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('phi-axis', fontweight='bold', fontsize=20)
    plt.ylabel('err-axis', fontweight='bold', fontsize=20)
    plt.xticks(np.arange(-180,181,10))

    plt.savefig('phi_err_bar.png')
    plt.close(fig2)

def err_polar_box(theta_list, phi_list, c):
    #每n度
    theta_tensor = torch.tensor(theta_list)
    c_tensor = torch.tensor(c)
    num=18
    deflaut_clip=math.pi/num
    avg_list = []
    std_list = []
    theta_value_list = []
    # i=1
    for i in range(num):
        theta_select = (theta_tensor>(i)*deflaut_clip) & (theta_tensor<(i+1)*deflaut_clip)
        theta_value = c_tensor[theta_select]
        avg = theta_value.mean()
        std = theta_value.std()
        avg_list.append(avg)
        std_list.append(std)
        theta_value_list.append(theta_value)


    fig1 = plt.figure(figsize=(12, 8),dpi=(600))

    # ax = plt.axes()
    # my_cmap = plt.get_cmap('rainbow')

    plt.boxplot(theta_value_list,showfliers=False,medianprops={'linestyle':'-','linewidth':1,'color':'black'},
                labels=[int(180*(i+0.5)/num) for i in range(num)])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('theta/deg', fontsize=20, fontweight='bold')
    plt.ylabel('Attitude Error/deg', fontsize=20, fontweight='bold')


    plt.savefig('theta_err_box.png')
    plt.close(fig1)
    #每30度phi
    phi_tensor = torch.tensor(phi_list)
    c_tensor = torch.tensor(c)
    num=18
    deflaut_clip=2*math.pi/num
    avg_list = []
    std_list = []
    phi_value_list = []
    # i=1
    for i in range(num):
        phi_select = (phi_tensor>(i)*deflaut_clip-math.pi) & (phi_tensor<(i+1)*deflaut_clip-math.pi)
        phi_value = c_tensor[phi_select]
        avg = phi_value.mean()
        std = phi_value.std()
        avg_list.append(avg)
        std_list.append(std)
        phi_value_list.append(phi_value)


    fig2 = plt.figure(figsize=(12, 8),dpi=(600))
    import matplotlib.ticker as ticker
    ax = plt.axes()
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    # my_cmap = plt.get_cmap('rainbow')
    plt.boxplot(phi_value_list,showfliers=False,medianprops={'linestyle':'-','linewidth':1,'color':'black'},
                labels=[int(360*(i+0.5)/num - 180) for i in range(num)])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('phi/deg', fontsize=20, fontweight='bold')
    plt.ylabel('Attitude Error/deg', fontsize=20, fontweight='bold')
    # plt.xticks(np.arange(-180,181,10))

    plt.savefig('phi_err_box.png')
    plt.close(fig2)


def err_distance_box(distance_list,c):
    #每n度
    distance_tensor = torch.tensor(distance_list)
    c_tensor = torch.tensor(c)
    num=20
    deflaut_clip=10/num
    avg_list = []
    std_list = []
    distance_value_list = []
    # i=1
    for i in range(num):
        if i < 4:
            continue
        distance_select = (distance_tensor>(i)*deflaut_clip) & (distance_tensor<(i+1)*deflaut_clip)
        distance_value = c_tensor[distance_select]
        avg = distance_value.mean()
        std = distance_value.std()
        avg_list.append(avg)
        std_list.append(std)
        distance_value_list.append(distance_value)


    fig = plt.figure(figsize=(9, 6),dpi=(600))
    ax = fig.add_subplot(111)
    # ax = plt.axes()
    # my_cmap = plt.get_cmap('rainbow')
    # , labels = [180 * (i + 0.5) / num for i in range(num)]
    plt.boxplot(distance_value_list,showfliers=False,widths=0.4,
                boxprops={'color': 'black',  # 箱子外框
                          },
                medianprops={'linestyle':'-','linewidth':1,'color':'black'},
                labels=[float(10*(i+4+0.5)/num) for i in range(num-4)])
    plt.xlabel('Target Relative Distance/m', fontsize=16, fontweight='bold')
    plt.ylabel('Attitude Error/deg', fontsize=16, fontweight='bold')
    plt.yscale('symlog', base=10)
    # ax.yaxis.minorticks_on()

    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(0.1))
    # from matplotlib.ticker import Locator
    # class LogMinorLocator(Locator):
    #     def __call__(self):
    #         majorlocs = self.axis.get_majorticklocs()
    #         step = majorlocs[1] - majorlocs[0]
    #         res = majorlocs[:, None] + np.log10(np.linspace(1, 0.1, 10)) * step
    #         return res.ravel()
    #
    # ax.yaxis.set_minor_locator(LogMinorLocator())

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.xlim(1,10)
    # import matplotlib.ticker as ticker
    # ax = plt.axes()
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

    plt.savefig('distance_err_box.png')
    plt.close(fig)

def err_kp_num_plot(kp_num_list,c):
    # 每n度
    distance_tensor = torch.tensor(kp_num_list)
    c_tensor = torch.tensor(c)
    kp_num_sum = torch.zeros(11)
    num = 11
    avg_list = []
    std_list = []
    distance_value_list = []
    # i=1
    for i in range(num):
        distance_select = distance_tensor == i
        distance_value = c_tensor[distance_select]
        avg = distance_value.mean()
        std = distance_value.std()
        avg_list.append(avg)
        std_list.append(std)
        distance_value_list.append(distance_value)

    for i in kp_num_list:
        kp_num_sum[i-1]+=1

    fig = plt.figure(dpi=(600))
    ax = fig.add_subplot(211)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.boxplot(distance_value_list, showfliers=False, widths=0.4,
                boxprops={'color': 'black',  # 箱子外框
                          },
                medianprops={'linestyle': '-', 'linewidth': 1, 'color': 'black'})
    ax.set_yscale('symlog',base=10)
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(0.1))
    ax2 = fig.add_subplot(212)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    for i in range(num):
        ax2.bar(i+1, kp_num_sum[i]*2)
    plt.xticks(np.arange(1,12,1))
    plt.subplots_adjust(top=0.9,bottom=0.12,hspace=0.3)
    plt.xlabel('Num. of Selected Key-points ', fontsize=16, fontweight='bold')
    ax.set_ylabel('Attitude Error/deg', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Num. of images', fontsize=16, fontweight='bold')



    # plt.xlim(1,10)
    # import matplotlib.ticker as ticker
    # ax = plt.axes()
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

    plt.savefig('kp_num.png')
    plt.close(fig)

import seaborn as sns
import pandas as pd
def get_distance_data(distance_list,c):
    #每n度
    distance_tensor = torch.tensor(distance_list)
    c_tensor = torch.tensor(c)
    num=20
    deflaut_clip=10/num
    avg_list = []
    std_list = []
    distance_value_list = []
    # i=1
    for i in range(num):
        if i < 4:
            continue
        distance_select = (distance_tensor>(i)*deflaut_clip) & (distance_tensor<(i+1)*deflaut_clip)
        distance_value = c_tensor[distance_select]
        avg = distance_value.mean()
        std = distance_value.std()
        avg_list.append(avg)
        std_list.append(std)
        distance_value_list.append(distance_value)
    # xx=torch.stack(distance_value_list)
    # print(xx)
    return [float(10*(i+4+0.5)/num) for i in range(num-4)],distance_value_list

def plot_data(data, xaxis='Epoch', value="TestEpRet",
              condition="Condition1", smooth=1,
              linewidth=4,
              rank=True,
              performance=True,
              style='Line',
              **kwargs):
    sns.set(style="darkgrid", font_scale=1.75, )
    # # # data按照lenged排序；
    # data.sort_values(by='Condition1', axis=0)
    # sns.colors_palette('Paired',6)
    # sns.color_palette("Paired")
    # palette = sns.xkcd_palette(["pinky", "neon red", "kiwi", "cool green", "sky blue", "bright blue"])
    palette = sns.color_palette("Paired", 6, )
    if style=='Line':
        g_results=sns.lineplot(
                    data=data,
                    x=xaxis,
                    y=value,
                    # hue=condition,
                    # errorbar=('ci',95),
                    err_style='bars',
                    errorbar=("se", 2),
                    linewidth=linewidth,
                    palette=palette,
                    # markers = ['','o','', 'o','', 'o'],
                    **kwargs)
    elif style=='plot':
        g_results=sns.scatterplot(
                    data=data,
                    x=xaxis,
                    y=value,
                    # linewidth=linewidth,
                    )
    # g_results.set(yscale='log')
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:
        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)
    Changes the colorscheme and the default legend style, though.        

    plt.legend()
        loc:图例位置,可取('best', 'upper right', 'upper left', 'lower left', 'lower right', 
            'right', 'center left', 'center , right', 'lower center', 'upper center', 'center')
            若是使用了bbox_to_anchor,则这项就无效了
        fontsize: int或float或{'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'},字体大小；
        frameon: 是否显示图例边框,
        ncol: 图例的列的数量,默认为1,
        title: 为图例添加标题
        shadow: 是否为图例边框添加阴影,
        markerfirst: True表示图例标签在句柄右侧,false反之,
        markerscale: 图例标记为原图标记中的多少倍大小,
        numpoints: 表示图例中的句柄上的标记点的个数,一般设为1,
        fancybox: 是否将图例框的边角设为圆形
        framealpha: 控制图例框的透明度
        borderpad: 图例框内边距
        labelspacing: 图例中条目之间的距离
        handlelength: 图例句柄的长度
        bbox_to_anchor: (横向看右,纵向看下),如果要自定义图例位置或者将图例画在坐标外边,用它,
            比如bbox_to_anchor=(1.4,0.8),这个一般配合着ax.get_position(),
            set_position([box.x0, box.y0, box.width*0.8 , box.height])使用
    """
    # # 对图例legend也做一个排序，这样看起来更直观~
    # handles, labels = plt.gca().get_legend_handles_labels()
    # plt.legend(handles, labels, labelspacing=0.2,
    #             ncol=1,
    #             handlelength=3,
    #            # mode="expand",
    #             borderaxespad=0.,
    #             loc = 'upper center',
    #            )

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:
    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout(pad=0.5)

def make_plots(data,name_str='temp', legend=None,
               xaxis=None, values=None,
               condition=None,
               font_scale=1.5, smooth=1,
               linewidth=3,
               select=None, exclude=None,
               estimator='mean',
               rank=True,
               performance=True,
               style='Line',
                **kwargs
               ):
    values = values if isinstance(values, list) else [values]
    estimator = getattr(np, estimator)  # choose what to show on main curve: mean? max? min?
    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value=value,
                  condition=condition, smooth=smooth, estimator=estimator,
                  linewidth=linewidth, rank=rank, performance=performance,style=style,**kwargs)
    # 默认最大化图片
    manager = plt.get_current_fig_manager()
    try:
        # matplotlib3.3.4 work
        manager.resize(*manager.window.maxsize())
    except:
        # matplotlib3.2.1//2.2.3 work
        manager.window.showMaximized()
    fig = plt.gcf()
    fig.set_size_inches((16, 9), forward=False)

    # select_str = ''
    # exclude_str = ''
    # print("select:", select)
    # print("select_str:", select_str)
    # if select is not None and type(select) is list:
    #     for s_str in select:
    #         select_str += s_str
    # if exclude is not None and type(exclude) is list:
    #     for s_str in exclude:
    #         exclude_str += s_str
    # print("select_str:", select_str)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    # plt.ylim(0, 1.03)
    fig.savefig(name_str+'.png',
                bbox_inches='tight',
                dpi=300)




def getdata(fn,IS_REFINE,split):
    distance_min = 0
    distance_max = 15
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    T = [
        ToTensorV2(p=1.0)
    ]  # transforms
    trans = A.Compose(T, keypoint_params=A.KeypointParams(format='xy',
                                                          remove_invisible=False))
    dataset_root_dir = '/opt/dl_workspace/datasets/speedplus/speedplus/'  # path to speed+'
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = PyTorchSatellitePoseEstimationDataset(split=split, speed_root=dataset_root_dir,
                                                      points=points, transform=None, DATAGENERATE=True)


    labels = train_set.get_labels()

    err_label_list=[]
    import json
    with open('./saveload/'+ fn + 'result.json', 'r') as f:
        err_label_list = json.load(f)
    x=[]
    y=[]
    z=[]
    c=[]
    d=[]
    e=[]
    f=[]
    theta_list = []
    phi_list = []
    kp_num_list = []

    pixel_location_x_list = []
    pixel_location_y_list = []

    distance_list = []
    distance_list2 = []
    few_shot_list=[]
    index_ = 0
    for errlabel in err_label_list:

        filename = errlabel['filename']
        err_ori = errlabel['err_ori']
        los_r = errlabel['los_r']
        label = labels[filename[0]]
        rgt = torch.tensor(label["r"],dtype=float)
        qgt = torch.tensor(label["q"])
        kp_num = errlabel['kp_num']
        padded_ratio = errlabel['padded_ratio']
        Rmats = quaternion2rot(qgt)

        # print(math.sqrt(sum(rgt*rgt)))
        if math.sqrt(sum(rgt*rgt))>distance_min and math.sqrt(sum(rgt*rgt)) <= distance_max:
            t = points_trans_accord(Rmats, rgt.view(3,-1)).squeeze()
            theta, phi = xyz2sphere(t)
            box = train_set.get_boxes_from_q_r(q=qgt, r=rgt)
            x1, y1, x2, y2 = box
            # if los_r > 1:
            #     index_+=1
            # if err_ori <= 180.0 and torch.norm(rgt) > 3.5:
            if err_ori <= 180.0:
                x.append(t[0])
                y.append(t[1])
                z.append(t[2])
                c.append(err_ori)
                d.append(los_r)
                e.append(math.sqrt(sum(rgt * rgt))*los_r)
                f.append(padded_ratio[0])
                theta_list.append(theta*180/math.pi)
                phi_list.append(phi*180/math.pi)
                kp_num_list.append(kp_num)
                pixel_location_x_list.append((x1 + x2) / 2 / 1960)
                pixel_location_y_list.append((y1 + y2) / 2 / 1200)
                distance = math.sqrt(sum(rgt * rgt))
                distance_list.append(distance)
                if IS_REFINE:
                    few_shot_list.append(fn[:-2]+'_refine')
                else:
                    few_shot_list.append(fn[:-2])
                x_distance=np.arange(2-0.25,10.25,0.5)

                if distance<2.25:
                    distance_list2.append(x_distance[0])
                elif distance < 2.5:
                    distance_list2.append(x_distance[1])
                elif distance<3:
                    distance_list2.append(x_distance[2])
                elif distance<3.5:
                    distance_list2.append(x_distance[3])
                elif distance<4:
                    distance_list2.append(x_distance[4])
                elif distance<4.5:
                    distance_list2.append(x_distance[5])
                elif distance<5:
                    distance_list2.append(x_distance[6])
                elif distance<5.5:
                    distance_list2.append(x_distance[7])
                elif distance<6:
                    distance_list2.append(x_distance[8])
                elif distance<6.5:
                    distance_list2.append(x_distance[9])
                elif distance<7:
                    distance_list2.append(x_distance[10])
                elif distance<7.5:
                    distance_list2.append(x_distance[11])
                elif distance<8:
                    distance_list2.append(x_distance[12])
                elif distance<8.5:
                    distance_list2.append(x_distance[13])
                elif distance<9:
                    distance_list2.append(x_distance[14])
                elif distance<9.5:
                    distance_list2.append(x_distance[15])
                elif distance<10:
                    distance_list2.append(x_distance[16])
                # elif distance<10:
                #     distance_list2.append(x_distance[17])

    pd_dict={'Attitude Error/deg':c,'los_r_relative':d,\
             'Num. of Selected Key-points':kp_num_list,\
             'Position Error/m':e,'theta_list':theta_list, 'phi_list':phi_list,\
             'distance_list':distance_list, 'Target Relative Distance/m': distance_list2,\
             'condition': few_shot_list, 'Truncation Ratio': f}
    print(fn,IS_REFINE)
    data = torch.tensor(c)
    print(split,'los_ori_avg:',data.mean())
    data = torch.tensor(d)
    print(split,'los_r_avg_relative:',data.mean())
    data = torch.tensor(e)
    print(split,'los_r_avg:',data.mean())
    return pd_dict

def get_data(target_result,IS_REFINE):
    pd_dict={}
    ff=0

    for fn, is_refine in zip(target_result,IS_REFINE):
        if fn[:7]=='sunlamp':
            split='sunlamp'
        elif fn[:8]=='lightbox':
            split = 'lightbox'
        else:
            split = 'validation'
        dict=getdata(fn+'/'+str(is_refine),is_refine,split)
        # pd_dict.update(dict)
        ff+=1
        if ff == 1:
            pd_dict=dict
        else:
            for k,v in dict.items():
                pd_dict[k]+=dict[k]
    return pd_dict

def make_proportion_and_err(pddata,name_str,cmap='warmup',y='Attitude Error/deg'):
    fig=plt.figure()
    fig.add_subplot(2,1,1)
    # ax=sns.countplot(data=pddata, x='Num. of select key-points')

    prop_df = (pddata['Num. of Selected Key-points']
               .value_counts(normalize=True)
               .rename("Proportion/%")
               .reset_index())
    prop_df["Proportion/%"]=prop_df["Proportion/%"]*100
    ax=sns.barplot(x='index', y="Proportion/%", data=prop_df)
    ax.set(xlabel=None)
    fig.add_subplot(2, 1, 2)
    sns.barplot(data=pddata,x='Num. of Selected Key-points',y=y,estimator=np.mean)

    fig = plt.gcf()
    fig.set_size_inches((16, 9), forward=False)
    fig.savefig(name_str+'.png',
                bbox_inches='tight',
                dpi=300)

def make_heatmap(pddata,x='phi_list',y='theta_list',c='Attitude Error/deg',cmap="coolwarm", gridsize=20,name_str='make_heatmap'):
    plt.figure()
    plt.hexbin(pddata[x],pddata[y], pddata[c],cmap=cmap,gridsize=gridsize)
    plt.colorbar()
    plt.xlabel('phi/deg')
    plt.ylabel('theta/deg')
    fig = plt.gcf()
    fig.set_size_inches((16, 9), forward=False)
    fig.savefig(name_str+'.png',
                bbox_inches='tight',
                dpi=300)

def plot_scatter(x, y, title='Scatter Plot', x_label='X Axis', y_label='Y Axis'):
    """
    该函数接收x和y坐标的列表，绘制一个散点图，并显示它。

    参数：
    - x: x坐标的列表。
    - y: y坐标的列表。
    - title: 图表的标题（默认为'Scatter Plot'）。
    - x_label: x轴的标签（默认为'X Axis'）。
    - y_label: y轴的标签（默认为'Y Axis'）。
    """

    # 确保x和y的长度相同
    if len(x) != len(y):
        raise ValueError("Input lists x and y must have the same length.")

    # 创建一个散点图
    plt.scatter(x, y)

    # 为图表添加标题和轴标签
    # plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(0.0, 0.5)
    # 显示图表
    plt.show()

if __name__ == '__main__':

    # target_result = ['synthetic_1026_few_shot','synthetic_1026_few_shot','synthetic_513_few_shot','synthetic_513_few_shot','synthetic_128_few_shot','synthetic_128_few_shot',\
    #                  'sunlamp_513_few_shot', 'sunlamp_513_few_shot', 'sunlamp_128_few_shot', 'sunlamp_128_few_shot', 'sunlamp_64_few_shot', 'sunlamp_64_few_shot',\
    #                  'lightbox_513_few_shot', 'lightbox_513_few_shot', 'lightbox_128_few_shot', 'lightbox_128_few_shot', 'lightbox_64_few_shot', 'lightbox_64_few_shot']
    #
    # IS_REFINE=[0,1,0,1,0,1]
    # linestyles=['-','--','-','--','-','--']
    # pd_dict=get_data(target_result[:6],IS_REFINE[:6])
    # pddata=pd.DataFrame(pd_dict)
    # make_plots(data=pddata,name_str='distance_orierr_synthetic',xaxis='Target Relative Distance/m',values='Attitude Error/deg',condition='condition')
    # make_plots(data=pddata, name_str='distance_los_r_synthetic', xaxis='Target Relative Distance/m', values='Position Error/m',
    #            condition='condition')
    #
    # pd_dict = get_data(np.array(target_result)[:6][[3]],  np.array(IS_REFINE)[[3]])
    # pddata = pd.DataFrame(pd_dict)
    # make_proportion_and_err(pddata,name_str='prop_orierr_synthetic')
    # make_heatmap(pddata,name_str='heatmap_orierr_synthetic')
    #
    #
    #
    # pd_dict=get_data(target_result[6:12],IS_REFINE)
    # pddata=pd.DataFrame(pd_dict)
    # make_plots(data=pddata,name_str='distance_orierr_sunlamp',xaxis='Target Relative Distance/m',values='Attitude Error/deg',condition='condition')
    # make_plots(data=pddata, name_str='distance_los_r_sunlamp', xaxis='Target Relative Distance/m', values='Position Error/m',
    #            condition='condition')
    # pd_dict = get_data(np.array(target_result)[6:12][[1]], np.array(IS_REFINE)[[1]])
    # pddata = pd.DataFrame(pd_dict)
    # make_proportion_and_err(pddata,name_str='prop_orierr_sunlamp')
    # make_heatmap(pddata,name_str='heatmap_orierr_sunlamp')
    #
    # pd_dict=get_data(target_result[-6:],IS_REFINE)
    # pddata=pd.DataFrame(pd_dict)
    # make_plots(data=pddata,name_str='distance_orierr_lightbox',xaxis='Target Relative Distance/m',values='Attitude Error/deg',condition='condition')
    # make_plots(data=pddata, name_str='distance_los_r_lightbox', xaxis='Target Relative Distance/m', values='Position Error/m',
    #            condition='condition')
    # pd_dict = get_data(np.array(target_result)[-6:][[1]], np.array(IS_REFINE)[-6:][[1]])
    # pddata = pd.DataFrame(pd_dict)
    # make_proportion_and_err(pddata,name_str='prop_orierr_lightbox')
    # make_heatmap(pddata,name_str='heatmap_orierr_lightbox')


    target_result_total = ['synthetic','sunlamp','lightbox']
    IS_REFINE_total=[0,0,0]
    target_result_total = ['synthetic']
    IS_REFINE_total=[0]
    pd_dict=get_data(target_result_total,IS_REFINE_total)
    pddata=pd.DataFrame(pd_dict)
    make_plots(data=pddata,name_str='distance_orierr_lightbox',xaxis='Target Relative Distance/m',values='Attitude Error/deg',condition='condition')
    make_plots(data=pddata, name_str='distance_los_r_lightbox', xaxis='Target Relative Distance/m', values='Position Error/m',condition='condition')
    make_plots(data=pddata[pddata['Truncation Ratio']>0.0], name_str='Truncation Ratio', \
               xaxis='Truncation Ratio', values='Attitude Error/deg',\
               condition='condition',style='plot')
    asdasd= pddata[pddata['Truncation Ratio'] > 0.0]
    print(pddata[pddata['Truncation Ratio'] > 0.0])
    x_value=[]
    y_value = []
    r_value = []
    full_image_y_value=[]
    full_image_r_value = []
    for ratio, err, r_err in zip(pd_dict['Truncation Ratio'], pd_dict['Attitude Error/deg'], pd_dict['Position Error/m']):
        if ratio != 0.0:
            x_value.append(ratio)
            y_value.append(err)
            r_value.append(r_err)
        else:
            full_image_y_value.append(err)
            full_image_r_value.append(r_err)
        if ratio > 1.0:
            print('error_ratio')
    # x_value = [_ != 0.0 for _ in pd_dict['padded_ratio']]
    ttt=torch.tensor(x_value).squeeze()
    print('attitude')
    print(f'ratio:{0.0} mean:', torch.tensor(full_image_y_value).mean())
    print(f'ratio:{0.0} std:', torch.tensor(full_image_y_value).std())
    print('los_r')
    print(f'ratio:{0.0} mean:', torch.tensor(full_image_r_value).mean())
    print(f'ratio:{0.0} std:', torch.tensor(full_image_r_value).std())
    for i in range(5):
        temp=[True if tt > i/10 and tt <= (i+1)/10 else False for tt in ttt]
        temp_atti=torch.tensor(temp).squeeze()
        print('attitude')
        print(f'ratio:{(i+1)/10} mean:',torch.tensor(y_value)[temp_atti].mean())
        print(f'ratio:{(i + 1) / 10} std:', torch.tensor(y_value)[temp_atti].std())
        print('los_r')
        print(f'ratio:{(i + 1) / 10} mean:', torch.tensor(r_value)[temp_atti].mean())
        print(f'ratio:{(i + 1) / 10} std:', torch.tensor(r_value)[temp_atti].std())

    print(torch.tensor(x_value).mean())
    print(torch.tensor(y_value).mean())
    print(torch.tensor(r_value).mean())
    plot_scatter(x_value, y_value, 'Example Scatter Plot', 'Truncation Ratio', 'Attitude Error/deg')
    # pd_dict = get_data(np.array(target_result_total)[[1]], np.array(IS_REFINE_total)[[1]])
    # pddata = pd.DataFrame(pd_dict)
    make_proportion_and_err(pddata,name_str='prop_orierr_total')
    make_heatmap(pddata,name_str='heatmap_orierr_total')
    # plt.show()

import matplotlib.pyplot as plt






