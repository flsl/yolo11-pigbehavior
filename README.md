# 基于yolov11的猪场猪只行为状态检测系统python源码+pytorch模型+评估指标曲线+精美GUI界面

【算法介绍】

在生猪养殖行业朝着智能化、精细化方向加速转型的关键时期，精准且高效地监测猪场猪只的行为状态，已成为保障养殖效益、提升动物福利以及防控疾病的核心挑战之一。猪场环境复杂多变，猪只的行为状态丰富多样，像日常的吃喝活动、嬉戏玩耍、休息睡眠，以及一些具有特殊意义的互动行为等，这些行为不仅直观反映了猪只当下的生理健康状况，更与养殖场的整体生产效率、疾病传播风险等紧密相连。一旦猪只出现异常行为，如长时间不进食、异常争斗、行动迟缓等，若未能及时察觉并采取相应措施，极易引发疾病大规模传播、生长速度减缓甚至猪只意外死亡，给养殖场带来巨大的经济损失。

传统猪场猪只行为状态检测方式主要依赖人工巡查。然而，受养殖场规模不断扩大、猪只数量日益增多以及空间布局复杂等因素限制，人工巡查很难全面覆盖猪场的各个区域，尤其是那些隐藏在猪群深处或处于角落位置的猪只状态，往往难以被及时观察到。而且，早期基于简单规则设定的监测方法，由于猪只毛发颜色各异、猪舍内光照条件不稳定以及各种设备、杂物遮挡等因素干扰，误判率高达 30%以上，根本无法满足养殖场“精准化、零疏漏”的管理需求。因此，开发一套具备高精度、强适应性且能实时监测的猪场猪只行为状态智能检测系统，成为提升养殖场管理水平和养殖效益的关键技术突破点。

目前现有技术存在诸多明显瓶颈：人工巡查不仅效率极其低下（单人单日仅能完成有限数量猪只的观察），而且巡查人员还面临着被猪只误伤、长时间工作导致观察疲劳等风险；基于颜色和简单形状分割的传统算法，难以准确区分猪只的正常行为与异常行为（例如，猪只正常休息时的蜷缩姿态与因疾病导致的不适蜷缩），在光线昏暗、猪舍内粉尘弥漫等低能见度环境下，算法性能会急剧下降；传统目标检测模型对猪只行为的多样性（如行走、奔跑、嬉戏、争斗等不同行为模式）和尺度变化（从幼猪的小巧灵活到成年猪的庞大笨重）适应性较差，对于小目标（如幼猪）的行为漏检率超过 40%，难以满足实际养殖场景的复杂需求。

基于 YOLOv11 的猪场猪只行为状态检测系统为生猪养殖管理带来了革命性的变革。YOLOv11 作为先进的目标检测算法，具备强大的特征提取和实时检测能力。该系统充分发挥 YOLOv11 的端到端实时检测优势，并针对猪场复杂环境进行了深度优化。

此系统能够精准识别猪只丰富多样的行为类别，具体涵盖：

-  **drink（饮水）** ：准确捕捉猪只饮水时的姿态和动作，判断其饮水是否正常，及时发现因饮水设备故障或健康问题导致的饮水异常情况。

-  **eat（进食）** ：实时监测猪只的进食行为，分析进食量、进食频率等数据，为评估猪只的食欲和营养状况提供依据。

-  **fight（争斗）** ：敏锐识别猪只之间的争斗行为，及时预警可能出现的受伤情况，避免争斗升级引发更严重的后果。

-  **investigating（探究）** ：察觉猪只对新环境、新物体或同伴的探究行为，了解猪只的好奇心和适应能力。

-  **jumpontopof（跳到……上面）** ：记录猪只这种较为特殊的行为，有助于分析猪只的活力和空间利用情况。

-  **lying（躺卧）** ：区分猪只正常的躺卧休息和因疾病、不适导致的异常躺卧，为猪只健康管理提供重要信息。

-  **nose - poke - elsewhere（用鼻子戳其他地方）** ：捕捉猪只这一细微行为，可能反映出猪只对周围环境的好奇或探索欲望。

-  **nose - to - nose（鼻对鼻）** ：识别猪只之间的这种互动行为，有助于了解猪只的社交关系和群体动态。

-  **other（其他）** ：作为一个兜底类别，可识别一些尚未明确分类的特殊行为，确保系统的全面性。

-  **playwithtoy（玩玩具）** ：监测猪只玩玩具的行为，评估猪只的娱乐需求和心理状态。

-  **run（奔跑）** ：记录猪只奔跑的行为，反映猪只的活力和运动能力。

-  **sitting（坐立）** ：观察猪只坐立的姿态和频率，辅助判断猪只的健康和舒适程度。

-  **sleep（睡眠）** ：准确识别猪只的睡眠状态，分析睡眠质量和时长，为评估猪只的休息状况提供数据支持。

-  **standing（站立）** ：监测猪只站立的行为，结合其他行为数据，全面了解猪只的活动模式。

-  **walk（行走）** ：实时跟踪猪只的行走轨迹和速度，评估猪只的运动能力和行动自由度。

通过对大量猪只行为状态图像数据的学习和训练，系统无论在白天还是夜晚，无论猪舍内光线如何变化，都能保持较高的检测准确率。同时，系统具备强大的抗干扰能力，能够有效应对猪舍内设备、粪便等干扰因素，为新型智能化养殖场建设提供了坚实的技术支撑，助力养殖场实现高效、精准、科学的养殖管理。

【效果展示】

<div style="text-align:center;"><img alt="" src="https://i-blog.csdnimg.cn/direct/3843dd2d62d8470cb4a2a0bf2d3a2d5f.jpeg"></div>

<div style="text-align:center;"><img alt="" src="https://i-blog.csdnimg.cn/direct/363029d2bf004d7b958f798f057bc8b7.jpeg">&nbsp;</div>

【测试环境】

windows10
anaconda3+python3.8
torch==2.3.0
ultralytics==8.3.81

【模型可以检测出类别】

```
drink
eat
fight
investigating
jumpontopof
lying
nose-poke-elsewhere
nose-to-nose
other
playwithtoy
run
sitting
sleep
standing
walk
```

【训练数据集介绍】

数据集格式：Pascal VOC格式+YOLO格式(不包含分割路径的txt文件，仅仅包含jpg图片以及对应的VOC格式xml文件和yolo格式txt文件)

图片数量(jpg文件个数)：3790

标注数量(xml文件个数)：3790

标注数量(txt文件个数)：3790

标注类别数：15

所在仓库：firc-dataset

标注类别名称(注意yolo格式类别顺序不和这个对应，而以labels文件夹classes.txt为准):["drink","eat","fight","investigating","jumpontopof","lying","nose-poke-elsewhere","nose-to-nose","other","playwithtoy","run","sitting","sleep","standing","walk"]

每个类别标注的框数：

drink 框数 = 409

eat 框数 = 3738

fight 框数 = 506

investigating 框数 = 6915

jumpontopof 框数 = 6

lying 框数 = 2760

nose-poke-elsewhere 框数 = 43

nose-to-nose 框数 = 551

other 框数 = 2

playwithtoy 框数 = 92

run 框数 = 108

sitting 框数 = 394

sleep 框数 = 8356

standing 框数 = 3783

walk 框数 = 2559

总框数：30222

使用标注工具：labelImg

标注规则：对类别进行画矩形框

重要说明：暂无

特别声明：本数据集不对训练的模型或者权重文件精度作任何保证，数据集只提供准确且合理标注

图片预览：

<img src="./assets/331_3.jpeg" alt="" style="max-height:736px; box-sizing:content-box;" />

标注例子：

<img src="./assets/331_4.jpeg" alt="" style="max-height:1920px; box-sizing:content-box;" />

【训练信息】

| 参数 | 值 |
|:---:|:---:|
| 训练集图片数 | 3600 |
| 验证集图片数 | 190 |
| 训练map | 87.9% |
| 训练精度(Precision) | 91.3% |
| 训练召回率(Recall) | 81.9% |

【验证集精度统计】

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| all | 190 | 1516 | 0.913 | 0.819 | 0.879 | 0.752 |
| drink | 15 | 15 | 1 | 0.985 | 0.995 | 0.88 |
| eat | 76 | 166 | 0.976 | 1 | 0.995 | 0.919 |
| fight | 17 | 27 | 0.947 | 0.963 | 0.977 | 0.764 |
| investigating | 142 | 369 | 0.941 | 0.949 | 0.984 | 0.825 |
| jumpontopof | 1 | 1 | 1 | 0 | 0 | 0 |
| lying | 55 | 139 | 0.952 | 0.998 | 0.994 | 0.948 |
| nose-poke-elsewhere | 1 | 1 | 0.862 | 1 | 0.995 | 0.796 |
| nose-to-nose | 18 | 33 | 0.959 | 0.97 | 0.983 | 0.893 |
| other | 1 | 1 | 1 | 0 | 0.995 | 0.895 |
| playwithtoy | 6 | 6 | 0.815 | 1 | 0.995 | 0.835 |
| run | 8 | 8 | 0.496 | 0.625 | 0.385 | 0.211 |
| sitting | 22 | 24 | 1 | 1 | 0.995 | 0.858 |
| sleep | 127 | 405 | 1 | 0.998 | 0.995 | 0.978 |
| standing | 68 | 158 | 0.875 | 0.924 | 0.976 | 0.845 |
| walk | 93 | 163 | 0.876 | 0.866 | 0.916 | 0.633 |

【界面设计】

```
class Ui_MainWindow(QtWidgets.QMainWindow):
    signal = QtCore.pyqtSignal(str, str)
 
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1280, 728)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
 
        self.weights_dir = './weights'
 
        self.picture = QtWidgets.QLabel(self.centralwidget)
        self.picture.setGeometry(QtCore.QRect(260, 10, 1010, 630))
        self.picture.setStyleSheet("background:black")
        self.picture.setObjectName("picture")
        self.picture.setScaledContents(True)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 81, 21))
        self.label_2.setObjectName("label_2")
        self.cb_weights = QtWidgets.QComboBox(self.centralwidget)
        self.cb_weights.setGeometry(QtCore.QRect(10, 40, 241, 21))
        self.cb_weights.setObjectName("cb_weights")
        self.cb_weights.currentIndexChanged.connect(self.cb_weights_changed)
 
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 70, 72, 21))
        self.label_3.setObjectName("label_3")
        self.hs_conf = QtWidgets.QSlider(self.centralwidget)
        self.hs_conf.setGeometry(QtCore.QRect(10, 100, 181, 22))
        self.hs_conf.setProperty("value", 25)
        self.hs_conf.setOrientation(QtCore.Qt.Horizontal)
        self.hs_conf.setObjectName("hs_conf")
        self.hs_conf.valueChanged.connect(self.conf_change)
        self.dsb_conf = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dsb_conf.setGeometry(QtCore.QRect(200, 100, 51, 22))
        self.dsb_conf.setMaximum(1.0)
        self.dsb_conf.setSingleStep(0.01)
        self.dsb_conf.setProperty("value", 0.25)
        self.dsb_conf.setObjectName("dsb_conf")
        self.dsb_conf.valueChanged.connect(self.dsb_conf_change)
        self.dsb_iou = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dsb_iou.setGeometry(QtCore.QRect(200, 160, 51, 22))
        self.dsb_iou.setMaximum(1.0)
        self.dsb_iou.setSingleStep(0.01)
        self.dsb_iou.setProperty("value", 0.45)
        self.dsb_iou.setObjectName("dsb_iou")
        self.dsb_iou.valueChanged.connect(self.dsb_iou_change)
        self.hs_iou = QtWidgets.QSlider(self.centralwidget)
        self.hs_iou.setGeometry(QtCore.QRect(10, 160, 181, 22))
        self.hs_iou.setProperty("value", 45)
        self.hs_iou.setOrientation(QtCore.Qt.Horizontal)
        self.hs_iou.setObjectName("hs_iou")
        self.hs_iou.valueChanged.connect(self.iou_change)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 130, 72, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 210, 72, 21))
        self.label_5.setObjectName("label_5")
        self.le_res = QtWidgets.QTextEdit(self.centralwidget)
        self.le_res.setGeometry(QtCore.QRect(10, 240, 241, 400))
        self.le_res.setObjectName("le_res")
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1110, 30))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(self)
        self.toolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toolBar.setObjectName("toolBar")
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionopenpic = QtWidgets.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionopenpic.setIcon(icon)
        self.actionopenpic.setObjectName("actionopenpic")
        self.actionopenpic.triggered.connect(self.open_image)
        self.action = QtWidgets.QAction(self)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action.setIcon(icon1)
        self.action.setObjectName("action")
        self.action.triggered.connect(self.open_video)
        self.action_2 = QtWidgets.QAction(self)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/images/3.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_2.setIcon(icon2)
        self.action_2.setObjectName("action_2")
        self.action_2.triggered.connect(self.open_camera)
 
        self.actionexit = QtWidgets.QAction(self)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/images/4.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionexit.setIcon(icon3)
        self.actionexit.setObjectName("actionexit")
        self.actionexit.triggered.connect(self.exit)
 
        self.toolBar.addAction(self.actionopenpic)
        self.toolBar.addAction(self.action)
        self.toolBar.addAction(self.action_2)
        self.toolBar.addAction(self.actionexit)
 
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)
        self.init_all()
```



【常用评估参数介绍】

在目标检测任务中，评估模型的性能是至关重要的。你提到的几个术语是评估模型性能的常用指标。下面是对这些术语的详细解释：

Class：
这通常指的是模型被设计用来检测的目标类别。例如，一个模型可能被训练来检测车辆、行人或动物等不同类别的对象。
Images：
表示验证集中的图片数量。验证集是用来评估模型性能的数据集，与训练集分开，以确保评估结果的公正性。
Instances：
在所有图片中目标对象的总数。这包括了所有类别对象的总和，例如，如果验证集包含100张图片，每张图片平均有5个目标对象，则Instances为500。
P（精确度Precision）：
精确度是模型预测为正样本的实例中，真正为正样本的比例。计算公式为：Precision = TP / (TP + FP)，其中TP表示真正例（True Positives），FP表示假正例（False Positives）。
R（召回率Recall）：
召回率是所有真正的正样本中被模型正确预测为正样本的比例。计算公式为：Recall = TP / (TP + FN)，其中FN表示假负例（False Negatives）。
mAP50：
表示在IoU（交并比）阈值为0.5时的平均精度（mean Average Precision）。IoU是衡量预测框和真实框重叠程度的指标。mAP是一个综合指标，考虑了精确度和召回率，用于评估模型在不同召回率水平上的性能。在IoU=0.5时，如果预测框与真实框的重叠程度达到或超过50%，则认为该预测是正确的。
mAP50-95：
表示在IoU从0.5到0.95（间隔0.05）的范围内，模型的平均精度。这是一个更严格的评估标准，要求预测框与真实框的重叠程度更高。在目标检测任务中，更高的IoU阈值意味着模型需要更准确地定位目标对象。mAP50-95的计算考虑了从宽松到严格的多个IoU阈值，因此能够更全面地评估模型的性能。
这些指标共同构成了评估目标检测模型性能的重要框架。通过比较不同模型在这些指标上的表现，可以判断哪个模型在实际应用中可能更有效。

【使用步骤】

使用步骤：
（1）首先根据官方框架ultralytics安装教程安装好yolov11环境，并安装好pyqt5
（2）切换到自己安装的yolo11环境后，并切换到源码目录，执行python main.py即可运行启动界面，进行相应的操作即可

【提供文件】

python源码
yolo11n.pt模型
训练的map,P,R曲线图(在weights\results.png)
测试图片（在test_img文件夹下面）

注意提供训练的数据集，请到mytxt.txt文件中找到地址