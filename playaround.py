# from typing import Dict, Optional, Tuple, List

# import flwr as fl
# import tensorflow as tf
import numpy as np
# from keras.models import Sequential
# from keras import datasets, layers, models
# from keras.utils import np_utils
# from cinic10_ds import get_train_ds, get_test_val_ds
# import tensorflow_datasets as tfds
# import random
# import os
# from model import create_model
# from model_ascent import create_model_ascent
# import pandas as pd
import matplotlib.pyplot as plt
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
# from ray.util.multiprocessing import Pool



# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# np.set_printoptions(threshold=np.inf)

################# PLOTS HERE ####################

## changed batch norm
title = "Dirichlet b=0.1"
fedvalb = [0.09761952608823776, 0.09761952608823776, 0.1082216426730156, 0.14702939987182617, 0.16143228113651276, 0.138227641582489, 0.1674334853887558, 0.12222444266080856, 0.20024004578590393, 0.22724545001983643, 0.2900580167770386, 0.26225244998931885, 0.36327266693115234, 0.24784956872463226, 0.3516703248023987, 0.3378675878047943, 0.33346667885780334, 0.32666534185409546, 0.2974594831466675, 0.35547110438346863, 0.398479700088501, 0.3914783000946045, 0.4006801247596741, 0.33906781673431396, 0.4484896957874298, 0.5445088744163513, 0.3616723418235779, 0.36787357926368713, 0.21064212918281555, 0.35907182097435, 0.3986797332763672, 0.4464893043041229, 0.39527904987335205, 0.4188837707042694, 0.3528705835342407, 0.5117023587226868, 0.42508500814437866, 0.3940788209438324, 0.4192838668823242, 0.36367273330688477, 0.3912782669067383, 0.3618723750114441, 0.4536907374858856, 0.4824965000152588, 0.35847169160842896, 0.3026605248451233, 0.3368673622608185, 0.39267852902412415, 0.5159031748771667, 0.39567914605140686, 0.47389477491378784, 0.35807162523269653, 0.503300666809082, 0.5217043161392212, 0.477295458316803, 0.42968595027923584, 0.5025004744529724, 0.4508901834487915, 0.5095018744468689, 0.3380676209926605]
fedavg = [0.14242848753929138, 0.20984196662902832, 0.16123224794864655, 0.23544709384441376, 0.16143228113651276, 0.09961992502212524, 0.2836567461490631, 0.1972394436597824, 0.23104621469974518, 0.1378275603055954, 0.09961992502212524, 0.24944989383220673, 0.3300660252571106, 0.2952590584754944, 0.18343669176101685, 0.23484696447849274, 0.1548309624195099, 0.20744149386882782, 0.26705342531204224, 0.27725544571876526, 0.18863773345947266, 0.20004001259803772, 0.2858571708202362, 0.330466091632843, 0.2660531997680664, 0.29965993762016296, 0.1650330126285553, 0.3032606542110443, 0.29205840826034546, 0.3406681418418884, 0.28405681252479553, 0.39727944135665894, 0.37767553329467773, 0.13682736456394196, 0.3470694124698639, 0.35787156224250793, 0.2956591248512268, 0.28705739974975586, 0.2556511163711548, 0.3424685001373291, 0.28705739974975586, 0.3002600371837616, 0.38347670435905457, 0.32346469163894653, 0.29645928740501404, 0.31446290016174316, 0.2984596788883209, 0.3936787247657776, 0.4302860498428345, 0.364472895860672, 0.34086817502975464, 0.26725345849990845, 0.2762552499771118, 0.2998599708080292, 0.33626726269721985, 0.3550710082054138, 0.30926185846328735, 0.3626725375652313, 0.251650333404541, 0.442688524723053]

## pre change batch norm

title = "label 4 and 5 deleted in 70% of clients"
# fedval = [0.18043608963489532, 0.3378675878047943, 0.3620724081993103, 0.3992798626422882, 0.4974994957447052, 0.5699139833450317, 0.5951189994812012, 0.565913200378418, 0.6399279832839966, 0.6863372921943665, 0.695539116859436, 0.6859371662139893, 0.6887377500534058, 0.7201440334320068, 0.6871374249458313, 0.7267453670501709, 0.7351470589637756, 0.7479496002197266, 0.7405481338500977, 0.7413482666015625, 0.7347469329833984, 0.7619524002075195, 0.7665532827377319, 0.7375475168228149, 0.752550482749939, 0.728945791721344, 0.7725545167922974, 0.7609521746635437, 0.7769553661346436, 0.7667533755302429, 0.7757551670074463, 0.7793558835983276, 0.7849569916725159, 0.7847569584846497, 0.7769553661346436, 0.7807561755180359, 0.789557933807373, 0.7823565006256104, 0.7929586172103882, 0.7857571244239807, 0.774354875087738, 0.7811562418937683, 0.7849569916725159, 0.7845569252967834, 0.7757551670074463, 0.7881576418876648, 0.7941588163375854, 0.7903580665588379, 0.7973594665527344, 0.7837567329406738, 0.7915583252906799, 0.7905580997467041, 0.7899580001831055, 0.7029405832290649, 0.7949590086936951, 0.7891578078269958, 0.7909581661224365, 0.8013602495193481, 0.8003600835800171, 0.7911582589149475]
# fedvalada = [0.19743949174880981, 0.32146430015563965, 0.403280645608902, 0.48389679193496704, 0.5397079586982727, 0.5851170420646667, 0.6087217330932617, 0.5811161994934082, 0.6591318249702454, 0.6867373585700989, 0.674934983253479, 0.7009401917457581, 0.708341658115387, 0.7063412666320801, 0.7123424410820007, 0.7267453670501709, 0.7157431244850159, 0.7125425338745117, 0.7085416913032532, 0.6571314334869385, 0.7501500248908997, 0.7501500248908997, 0.7663532495498657, 0.7633526921272278, 0.764552891254425, 0.7673534750938416, 0.7685537338256836, 0.7691538333892822, 0.7733546495437622, 0.7735546827316284, 0.7801560163497925, 0.768953800201416, 0.7817563414573669, 0.7787557244300842, 0.7719544172286987, 0.7815563082695007, 0.784156858921051, 0.78395676612854, 0.7927585244178772, 0.7795559167861938, 0.7021404504776001, 0.7897579669952393, 0.7809562087059021, 0.7941588163375854, 0.7861572504043579, 0.7943588495254517, 0.7853570580482483, 0.7787557244300842, 0.7941588163375854, 0.7949590086936951, 0.7899580001831055, 0.7719544172286987, 0.7865573167800903, 0.8001600503921509, 0.7921584248542786, 0.7945588827133179, 0.7977595329284668, 0.8019604086875916, 0.7727545499801636, 0.7989597916603088]
fedvalada2 = [0.16843368113040924, 0.3616723418235779, 0.42988598346710205, 0.4804961085319519, 0.5143028497695923, 0.5805160999298096, 0.5531105995178223, 0.590718150138855, 0.6545308828353882, 0.6485297083854675, 0.6659331917762756, 0.6977395415306091, 0.6775355339050293, 0.7119423747062683, 0.7087417244911194, 0.7321464419364929, 0.7339468002319336, 0.7417483329772949, 0.7423484921455383, 0.7333466410636902, 0.7405481338500977, 0.7407481670379639, 0.7551510334014893, 0.763552725315094, 0.763552725315094, 0.7595518827438354, 0.758151650428772, 0.7643528580665588, 0.7717543244361877, 0.7541508078575134, 0.7607521414756775, 0.7655531167984009, 0.7559511661529541, 0.7607521414756775, 0.7723544836044312, 0.7779555916786194, 0.7817563414573669, 0.7755551338195801, 0.7457491755485535, 0.7875574827194214, 0.7805560827255249, 0.7819563746452332, 0.7807561755180359, 0.7627525329589844, 0.7801560163497925, 0.7851570248603821, 0.782956600189209, 0.7853570580482483, 0.7749549746513367, 0.7027405500411987, 0.7805560827255249, 0.7849569916725159, 0.7887577414512634, 0.7889577746391296, 0.7983596920967102, 0.7931586503982544, 0.7923584580421448, 0.7875574827194214, 0.7909581661224365, 0.7915583252906799]
fedavg = [0.2688537836074829, 0.38827764987945557, 0.41228246688842773, 0.4332866668701172, 0.5017003417015076, 0.5571114420890808, 0.5523104667663574, 0.5911182165145874, 0.6007201671600342, 0.622124433517456, 0.6213242411613464, 0.6459291577339172, 0.6309261918067932, 0.6473294496536255, 0.6483296751976013, 0.670734167098999, 0.6613322496414185, 0.6515303254127502, 0.6593318581581116, 0.6627325415611267, 0.6773354411125183, 0.6643328666687012, 0.6633326411247253, 0.6735346913337708, 0.6663332581520081, 0.670734167098999, 0.6699339747428894, 0.6689338088035583, 0.6657331585884094, 0.6681336164474487, 0.6865373253822327, 0.6899380087852478, 0.6763352751731873, 0.6755350828170776, 0.6787357330322266, 0.673734724521637, 0.690138041973114, 0.6773354411125183, 0.6817363500595093, 0.6807361245155334, 0.6905381083488464, 0.6947389245033264, 0.6897379755973816, 0.7257451415061951, 0.6809361577033997, 0.6817363500595093, 0.6957391500473022, 0.6829366087913513, 0.6825364828109741, 0.6865373253822327, 0.6833366751670837, 0.6851370334625244, 0.6829366087913513, 0.684736967086792, 0.6937387585639954, 0.6911382079124451, 0.6905381083488464, 0.6929385662078857, 0.693138599395752, 0.6893378496170044]
fedprox1 = [0.24064813554286957, 0.39067813754081726, 0.4230846166610718, 0.4844968914985657, 0.5187037587165833, 0.5451090335845947, 0.5623124837875366, 0.5745149254798889, 0.596119225025177, 0.6105220913887024, 0.6249249577522278, 0.6245248913764954, 0.6465293169021606, 0.645729124546051, 0.6541308164596558, 0.6559311747550964, 0.6671334505081177, 0.664132833480835, 0.6639328002929688, 0.6623324751853943, 0.6599319577217102, 0.665333092212677, 0.674934983253479, 0.6751350164413452, 0.676135241985321, 0.7119423747062683, 0.6775355339050293, 0.6755350828170776, 0.6917383670806885, 0.6875374913215637, 0.6725345253944397, 0.6795359253883362, 0.6751350164413452, 0.6805360913276672, 0.6797359585762024, 0.6807361245155334, 0.6817363500595093, 0.6833366751670837, 0.6871374249458313, 0.6917383670806885, 0.6879376173019409, 0.6833366751670837, 0.6871374249458313, 0.7055411338806152, 0.6817363500595093, 0.695539116859436, 0.6937387585639954, 0.7189437747001648, 0.6917383670806885, 0.6927385330200195, 0.7069413661956787, 0.6989398002624512, 0.6961392164230347, 0.6911382079124451, 0.695539116859436, 0.6925384998321533, 0.6947389245033264, 0.6919384002685547, 0.7221444249153137, 0.6961392164230347]

# title = "Dirichlet b=0.1"
# # fedval = [0.10022004693746567, 0.09761952608823776, 0.10262052714824677, 0.19863972067832947, 0.09981996566057205, 0.16183236241340637, 0.10322064161300659, 0.24584917724132538, 0.1796359270811081, 0.17803560197353363, 0.21344268321990967, 0.1970394104719162, 0.22824564576148987, 0.25505101680755615, 0.23464693129062653, 0.25205039978027344, 0.21584317088127136, 0.11122224479913712, 0.1554310917854309, 0.2020404040813446, 0.3722744584083557, 0.20484097301959991, 0.4340868294239044, 0.20384076237678528, 0.45689138770103455, 0.30626124143600464, 0.3820764124393463, 0.3722744584083557, 0.32166433334350586, 0.3524704873561859, 0.5217043161392212, 0.35887178778648376, 0.42008402943611145, 0.24684937298297882, 0.3350670039653778, 0.2936587333679199, 0.26265251636505127, 0.3792758584022522, 0.4058811664581299, 0.43728744983673096, 0.47149428725242615, 0.6315262913703918, 0.45689138770103455, 0.3916783332824707, 0.2836567461490631, 0.4404881000518799, 0.22364473342895508, 0.3274655044078827, 0.4502900540828705, 0.5497099161148071, 0.5311062335968018, 0.5993198752403259, 0.5787157416343689, 0.5501100420951843, 0.6097219586372375, 0.39287856221199036, 0.5819163918495178, 0.6119223833084106, 0.5093018412590027, 0.4332866668701172]
# # fedvalada = [0.14682936668395996, 0.10302060097455978, 0.10242048650979996, 0.10242048650979996, 0.10262052714824677, 0.13482695817947388, 0.0986197218298912, 0.16103219985961914, 0.09881976246833801, 0.13102620840072632, 0.2094418853521347, 0.10242048650979996, 0.2786557376384735, 0.20804160833358765, 0.1698339730501175, 0.15903180837631226, 0.32186436653137207, 0.21984396874904633, 0.2854571044445038, 0.3012602627277374, 0.33346667885780334, 0.23284657299518585, 0.2832566499710083, 0.3886777460575104, 0.290458083152771, 0.3946789503097534, 0.33106622099876404, 0.2244448959827423, 0.24564912915229797, 0.4014802873134613, 0.37047410011291504, 0.37007400393486023, 0.31926384568214417, 0.4352870583534241, 0.28705739974975586, 0.4746949374675751, 0.4042808413505554, 0.30146029591560364, 0.23064613342285156, 0.3818763792514801, 0.3076615333557129, 0.48169633746147156, 0.5457091331481934, 0.5289058089256287, 0.36287257075309753, 0.4214842915534973, 0.525905191898346, 0.4480896294116974, 0.5401080250740051, 0.5721144080162048, 0.49409881234169006, 0.5311062335968018, 0.551910400390625, 0.5245048999786377, 0.5239048004150391, 0.5911182165145874, 0.5509101748466492, 0.6039207577705383, 0.5383076667785645, 0.34806960821151733]
# # fedvalada2 = [0.14442887902259827, 0.10722144693136215, 0.1034206822514534, 0.09921984374523163, 0.23204641044139862, 0.10522104054689407, 0.1034206822514534, 0.15683136880397797, 0.14482896029949188, 0.17043408751487732, 0.17543508112430573, 0.19823965430259705, 0.21864372491836548, 0.3970794081687927, 0.2530505955219269, 0.43228647112846375, 0.27665531635284424, 0.3002600371837616, 0.3206641376018524, 0.37287458777427673, 0.3964793086051941, 0.5067013502120972, 0.31166234612464905, 0.29205840826034546, 0.35107022523880005, 0.36627325415611267, 0.4182836711406708, 0.4514903128147125, 0.305661141872406, 0.3666733205318451, 0.462692528963089, 0.4240848124027252, 0.38987797498703003, 0.5481096506118774, 0.418483704328537, 0.35147029161453247, 0.35107022523880005, 0.2568513751029968, 0.325665146112442, 0.32126426696777344, 0.5477095246315002, 0.4338867664337158, 0.45509102940559387, 0.518303632736206, 0.4710942208766937, 0.468093603849411, 0.4012802541255951, 0.4414882957935333, 0.4776955246925354, 0.48849770426750183, 0.42748549580574036, 0.4532906711101532, 0.4552910625934601, 0.3868773877620697, 0.5331066250801086, 0.546509325504303, 0.4530906081199646, 0.48609721660614014, 0.5773154497146606, 0.5537107586860657]
# # fedvalada4 = [0.16903381049633026, 0.10022004693746567, 0.1480296105146408, 0.09881976246833801, 0.1798359602689743, 0.18903781473636627, 0.2580516040325165, 0.17683537304401398, 0.18583716452121735, 0.15103021264076233, 0.18543708324432373, 0.26025205850601196, 0.231646329164505, 0.22204440832138062, 0.4020804166793823, 0.3368673622608185, 0.3946789503097534, 0.39727944135665894, 0.37787556648254395, 0.2666533291339874, 0.39827966690063477, 0.4290858209133148, 0.36807361245155334, 0.3422684669494629, 0.3642728626728058, 0.24584917724132538, 0.33326664566993713, 0.39007800817489624, 0.3964793086051941, 0.3816763460636139, 0.2740548253059387, 0.3624725043773651, 0.39027804136276245, 0.5159031748771667, 0.4874975085258484, 0.44428884983062744, 0.39587917923927307, 0.46709340810775757, 0.46169233322143555, 0.4806961417198181, 0.5541108250617981, 0.5241048336029053, 0.5165032744407654, 0.34546908736228943, 0.3942788541316986, 0.3696739375591278, 0.4894979000091553, 0.43208640813827515, 0.4798959791660309, 0.4066813290119171, 0.40088018774986267, 0.3812762498855591, 0.26285257935523987, 0.42928585410118103, 0.26285257935523987, 0.41768354177474976, 0.42448490858078003, 0.45249050855636597, 0.46929386258125305, 0.5001000165939331]
# fedvalada5 = [0.12642528116703033, 0.11182236671447754, 0.14142829179763794, 0.1008201614022255, 0.17103420197963715, 0.1654330939054489, 0.1700340062379837, 0.2090418040752411, 0.13202640414237976, 0.1726345270872116, 0.12602519989013672, 0.18583716452121735, 0.19363872706890106, 0.22104421257972717, 0.19583916664123535, 0.138227641582489, 0.22944588959217072, 0.13442689180374146, 0.1746349334716797, 0.1992398500442505, 0.22704540193080902, 0.23524704575538635, 0.26465293765068054, 0.49369874596595764, 0.43468692898750305, 0.37787556648254395, 0.4334867000579834, 0.4900980293750763, 0.3302660584449768, 0.19783957302570343, 0.22964592278003693, 0.18343669176101685, 0.35567113757133484, 0.5089017748832703, 0.3798759877681732, 0.29725944995880127, 0.4734947085380554, 0.3714742958545685, 0.20124024152755737, 0.3944788873195648, 0.4068813621997833, 0.36087217926979065, 0.2906581461429596, 0.2606521248817444, 0.41248250007629395, 0.4808961749076843, 0.38547709584236145, 0.4086817502975464, 0.2460492104291916, 0.43768754601478577, 0.39567914605140686, 0.4606921374797821, 0.43248650431632996, 0.3866773247718811, 0.389277845621109, 0.47149428725242615, 0.41528305411338806, 0.4756951332092285, 0.535507082939148, 0.5749149918556213]
# # mm = [0.0986197218298912, 0.10682136565446854, 0.10862172394990921, 0.10242048650979996, 0.10162032395601273, 0.09941988438367844, 0.1010202020406723, 0.1008201614022255, 0.12682536244392395, 0.09901980310678482, 0.19403880834579468, 0.25985196232795715, 0.3106621205806732, 0.2044408917427063, 0.20624125003814697, 0.3716743290424347, 0.4634926915168762, 0.4260852038860321, 0.2534506916999817, 0.3542708456516266, 0.3300660252571106, 0.37107422947883606, 0.37287458777427673, 0.4088817834854126, 0.4064812958240509, 0.42508500814437866, 0.4410882294178009, 0.5363072752952576, 0.4556911289691925, 0.39307862520217896, 0.3472694456577301, 0.36607322096824646, 0.43728744983673096, 0.506501317024231, 0.5417083501815796, 0.4580916166305542, 0.5003000497817993, 0.4684937000274658, 0.4630926251411438, 0.477295458316803, 0.5915182828903198, 0.546509325504303, 0.5683136582374573, 0.5265052914619446, 0.4206841289997101, 0.4658931791782379, 0.477895587682724, 0.53670734167099, 0.6083216667175293, 0.5979195833206177, 0.6309261918067932, 0.6037207245826721, 0.5957191586494446, 0.560312032699585, 0.5695139169692993, 0.586317241191864, 0.542108416557312, 0.5797159671783447, 0.526905357837677, 0.5343068838119507]
# fedavg = [0.10182036459445953, 0.10182036459445953, 0.10562112182378769, 0.15463092923164368, 0.17903581261634827, 0.1278255581855774, 0.14982996881008148, 0.18883776664733887, 0.2018403708934784, 0.2142428457736969, 0.22224444150924683, 0.19823965430259705, 0.19843968749046326, 0.20604120194911957, 0.36287257075309753, 0.09761952608823776, 0.28405681252479553, 0.21844369173049927, 0.17383477091789246, 0.27425485849380493, 0.11302260309457779, 0.2164432853460312, 0.2876575291156769, 0.1894378811120987, 0.13002599775791168, 0.25965192914009094, 0.3942788541316986, 0.29445889592170715, 0.1254250854253769, 0.22484496235847473, 0.24984997510910034, 0.33906781673431396, 0.31406280398368835, 0.29425886273384094, 0.369273841381073, 0.26725345849990845, 0.2580516040325165, 0.3550710082054138, 0.36107221245765686, 0.10182036459445953, 0.30466094613075256, 0.330466091632843, 0.37267452478408813, 0.12602519989013672, 0.3748749792575836, 0.35047009587287903, 0.49909982085227966, 0.39747950434684753, 0.2514503002166748, 0.30166032910346985, 0.11882376670837402, 0.1946389228105545, 0.4382876455783844, 0.3346669375896454, 0.14842969179153442, 0.4430886209011078, 0.4476895332336426, 0.43268653750419617, 0.37327465415000916, 0.34386876225471497]
# fedprox0001 = [0.19103820621967316, 0.14962992072105408, 0.10182036459445953, 0.1010202020406723, 0.09901980310678482, 0.09901980310678482, 0.21084216237068176, 0.19003801047801971, 0.1010202020406723, 0.10182036459445953, 0.1576315313577652, 0.10282056778669357, 0.18563713133335114, 0.11002200096845627, 0.12942588329315186, 0.19263853132724762, 0.2412482500076294, 0.11862372606992722, 0.17063412070274353, 0.1996399313211441, 0.2884576916694641, 0.18023604154586792, 0.20044009387493134, 0.15843167901039124, 0.2506501376628876, 0.24824965000152588, 0.10202040523290634, 0.15123024582862854, 0.3200640082359314, 0.25745150446891785, 0.20264053344726562, 0.38827764987945557, 0.2288457751274109, 0.3174634873867035, 0.2864573001861572, 0.16123224794864655, 0.25285056233406067, 0.398479700088501, 0.32406482100486755, 0.30646130442619324, 0.24104821681976318, 0.4660932123661041, 0.2486497312784195, 0.3122624456882477, 0.14382876455783844, 0.1156231239438057, 0.24004800617694855, 0.26425284147262573, 0.26245248317718506, 0.39487898349761963, 0.19303861260414124, 0.325065016746521, 0.1996399313211441, 0.2858571708202362, 0.2812562584877014, 0.47709542512893677, 0.28625723719596863, 0.3548709750175476, 0.4968993663787842, 0.33866772055625916]
# # fedprox1 = [0.10802160203456879, 0.10242048650979996, 0.10142028331756592, 0.10062012076377869, 0.15723145008087158, 0.15443088114261627, 0.13722744584083557, 0.305061012506485, 0.211642324924469, 0.10422084480524063, 0.2440488040447235, 0.09801960736513138, 0.24544909596443176, 0.12362472712993622, 0.2164432853460312, 0.12222444266080856, 0.2538507580757141, 0.16303260624408722, 0.09761952608823776, 0.2388477623462677, 0.1750349998474121, 0.21744349598884583, 0.19563913345336914, 0.17443488538265228, 0.14622925221920013, 0.17843568325042725, 0.09921984374523163, 0.15383076667785645, 0.1800360083580017, 0.10242048650979996, 0.21724344789981842, 0.19243848323822021, 0.1060212031006813, 0.1774354875087738, 0.16363272070884705, 0.17683537304401398, 0.2190438061952591, 0.1698339730501175, 0.16423285007476807, 0.37007400393486023, 0.36807361245155334, 0.30666133761405945, 0.18523705005645752, 0.26445290446281433, 0.2340468019247055, 0.19083817303180695, 0.09761952608823776, 0.11162232607603073, 0.42488497495651245, 0.201640322804451, 0.12002400308847427, 0.21324265003204346, 0.2090418040752411, 0.25705140829086304, 0.14822964370250702, 0.18803760409355164, 0.2580516040325165, 0.33126625418663025, 0.30446088314056396, 0.24244849383831024]
# # fedprox001 = [0.09881976246833801, 0.10242048650979996, 0.10502100735902786, 0.19863972067832947, 0.10262052714824677, 0.10302060097455978, 0.09761952608823776, 0.13482695817947388, 0.10962192714214325, 0.10202040523290634, 0.1380276083946228, 0.10362072288990021, 0.13002599775791168, 0.12022404372692108, 0.09921984374523163, 0.23544709384441376, 0.15203040838241577, 0.12222444266080856, 0.25525104999542236, 0.30626124143600464, 0.18523705005645752, 0.25505101680755615, 0.18043608963489532, 0.2836567461490631, 0.24264852702617645, 0.10062012076377869, 0.20264053344726562, 0.12602519989013672, 0.23084616661071777, 0.32126426696777344, 0.2490498125553131, 0.21144229173660278, 0.3156631290912628, 0.4380876123905182, 0.1746349334716797, 0.3224644958972931, 0.3518703877925873, 0.4234846830368042, 0.1230246052145958, 0.20144028961658478, 0.2586517333984375, 0.32886576652526855, 0.3786757290363312, 0.26245248317718506, 0.11522304266691208, 0.12662532925605774, 0.15463092923164368, 0.3594718873500824, 0.1622324436903, 0.1920384019613266, 0.2266453355550766, 0.27965593338012695, 0.3258651793003082, 0.3058611750602722, 0.18603721261024475, 0.2066413313150406, 0.4070814251899719, 0.4010802209377289, 0.45729145407676697, 0.35887178778648376]

# # print(sum(fedval))
# # print(sum(fedvalada))
# # print(sum(fedvalada2))
# # print(sum(mm))

# # title = "Dirichlet b=0.5"
# # fedval = [0.1800360083580017, 0.30686137080192566, 0.28905782103538513, 0.2708541750907898, 0.4776955246925354, 0.38827764987945557, 0.5757151246070862, 0.49889978766441345, 0.5931186079978943, 0.5127025246620178, 0.5211042165756226, 0.48169633746147156, 0.5419083833694458, 0.6463292837142944, 0.6233246922492981, 0.6487297415733337, 0.6599319577217102, 0.6547309756278992, 0.7153430581092834, 0.6421284079551697, 0.7035406827926636, 0.6895378828048706, 0.7155430912971497, 0.7293458580970764, 0.6825364828109741, 0.709541916847229, 0.6631326079368591, 0.710742175579071, 0.7641528248786926, 0.7551510334014893, 0.7459492087364197, 0.6815363168716431, 0.7505500912666321, 0.6091217994689941, 0.7487497329711914, 0.7535507082939148, 0.7175434827804565, 0.7091418504714966, 0.7381476163864136, 0.7811562418937683, 0.7569513916969299, 0.734346866607666, 0.7795559167861938, 0.7791558504104614, 0.7685537338256836, 0.7679535746574402, 0.7599520087242126, 0.7595518827438354, 0.7501500248908997, 0.7859572172164917, 0.774354875087738, 0.7879576086997986, 0.719143807888031, 0.7727545499801636, 0.7803560495376587, 0.713742733001709, 0.7565513253211975, 0.7697539329528809, 0.784156858921051, 0.7783556580543518]
# # fedavg = [0.08461692184209824, 0.10022004693746567, 0.21964393556118011, 0.2974594831466675, 0.3618723750114441, 0.4258851706981659, 0.5565112829208374, 0.4462892711162567, 0.47889578342437744, 0.5909181833267212, 0.6169233918190002, 0.4852970540523529, 0.5297059416770935, 0.4400880038738251, 0.5123024582862854, 0.5757151246070862, 0.582116425037384, 0.5593118667602539, 0.5585116744041443, 0.6141228079795837, 0.6399279832839966, 0.7161432504653931, 0.6487297415733337, 0.6551310420036316, 0.6603320837020874, 0.6323264837265015, 0.6413282752037048, 0.7219443917274475, 0.6933386921882629, 0.6891378164291382, 0.7153430581092834, 0.5935186743736267, 0.6355271339416504, 0.6279255747795105, 0.7183436751365662, 0.7523504495620728, 0.7015402913093567, 0.7099419832229614, 0.7477495670318604, 0.7561512589454651, 0.7587517499923706, 0.6961392164230347, 0.660932183265686, 0.7055411338806152, 0.7037407755851746, 0.616723358631134, 0.6335266828536987, 0.7751550078392029, 0.679135799407959, 0.6973394751548767, 0.7179436087608337, 0.7453490495681763, 0.7513502836227417, 0.7459492087364197, 0.7631526589393616, 0.7379475831985474, 0.6615322828292847, 0.6863372921943665, 0.7719544172286987, 0.7163432836532593, 0.6629325747489929]
# # fedprox001 = [0.11002200096845627, 0.16083216667175293, 0.27185437083244324, 0.23244649171829224, 0.42288458347320557, 0.4598919749259949, 0.3910782039165497, 0.4212842583656311, 0.518303632736206, 0.4086817502975464, 0.6269254088401794, 0.5339067578315735, 0.4676935374736786, 0.5695139169692993, 0.6421284079551697, 0.5937187671661377, 0.6655331254005432, 0.7075415253639221, 0.7033406496047974, 0.7269454002380371, 0.5111021995544434, 0.7009401917457581, 0.7303460836410522, 0.7241448163986206, 0.6887377500534058, 0.7675535082817078, 0.6537307500839233, 0.744148850440979, 0.7449489831924438, 0.7567513585090637, 0.7229446172714233, 0.6755350828170776, 0.7413482666015625, 0.7351470589637756, 0.665132999420166, 0.7425485253334045, 0.68353670835495, 0.7059412002563477, 0.7237447500228882, 0.778555691242218, 0.7669534087181091, 0.7701540589332581, 0.6919384002685547, 0.6999399662017822, 0.7451490163803101, 0.759151816368103, 0.7601520419120789, 0.6291258335113525, 0.7347469329833984, 0.7697539329528809, 0.7735546827316284, 0.7469493746757507, 0.7195439338684082, 0.7719544172286987, 0.7505500912666321, 0.6967393755912781, 0.7965593338012695, 0.7209441661834717, 0.6825364828109741, 0.7267453670501709]
# # fedvalprox = []
# # krum = [0.12982596457004547, 0.13422684371471405, 0.3342668414115906, 0.34646928310394287, 0.3688737750053406, 0.4266853332519531, 0.503300666809082, 0.5575115084648132, 0.5347069501876831, 0.5677135586738586, 0.5255051255226135, 0.5675135254859924, 0.5569114089012146, 0.4478895664215088, 0.4380876123905182, 0.6081216335296631, 0.6475294828414917, 0.6867373585700989, 0.6575314998626709, 0.651130199432373, 0.6009202003479004, 0.6315262913703918, 0.5923184752464294, 0.6993398666381836, 0.7489497661590576, 0.7063412666320801, 0.7061412334442139, 0.7271454334259033, 0.6693338751792908, 0.6527305245399475, 0.7583516836166382, 0.713742733001709, 0.6751350164413452, 0.5873174667358398, 0.7449489831924438, 0.6959391832351685, 0.7625524997711182, 0.7513502836227417, 0.6975395083427429, 0.6105220913887024, 0.7679535746574402, 0.7393478751182556, 0.7339468002319336, 0.7111422419548035, 0.7033406496047974, 0.727745532989502, 0.7547509670257568, 0.6129226088523865, 0.7121424078941345, 0.6747349500656128, 0.7001399993896484, 0.7631526589393616, 0.6865373253822327, 0.7125425338745117, 0.6621324419975281, 0.7433486580848694, 0.7657531499862671, 0.7831566333770752, 0.752550482749939, 0.7401480078697205]
# # # print(sum(fedprox0001))
# print(sum(fedvalada5))
# print(sum(fedavg))
# print(sum(fedprox0001))
# #print(sum(krum))

# # PLOT THE PLOT

# plt.plot(mm,label="fedvalada")
plt.plot(fedvalada2,label="fedval")
plt.plot(fedavg,label="fedavg")
#plt.plot(krum,label="mkrum")
plt.plot(fedprox1,label="fedprox u=0.01")
plt.legend(loc='upper left')
plt.title(title)
plt.show()

################# END PLOTS!! ####################

# x = np.array([[2,4],[5,6]])
# y = np.array([[2,1],[1,-3]])
# print(np.divide(x,y))
# w = np.unravel_index(abs(x).argmax(), x.shape)
# print(x[w])

# m = create_model("cifar10")
# x=m.get_weights()

# print(np.sum([np.prod(list(v.shape)) for v in x]))
# x = np.array([[2,10],[2,-4]])
# y = np.array([[0,1],[1,1]])
# z = np.unravel_index(abs(x).argmax(), x.shape)

# s = "1234"
# for i in range(len(s)):
#     if i+1 < len(s):
#         print(s[i+1])

#print(np.sum([np.prod(list(v.shape)) for v in x]))

# y = model.get_weights()

# w = fl.common.ndarrays_to_parameters(x)
# print(type(w))
# z = fl.common.parameters_to_ndarrays(w)
# print(type(z))
# x = np.array([[1,1],[1,1]])
# x = x/2
# print(x)
# print(x[0])
# for i in range(len(x)):
#     z.append(np.subtract(x[i],y[i]))

#print(np.unravel_index(x[0].argmax(), x[0].shape))
# i = np.unravel_index(x[0].argmax(), x[0].shape)
# print(x[0][i])
# x[0][i] = 0
# print(np.unravel_index(x[0].argmax(), x[0].shape))

#x[0][i] = -10
#print(x[0][i])
#print(np.argmax(x[0]))


# if __name__ == '__main__':
#     m = TesterClass("emnist")
#     m.the_test()

# def fun(x,y,n):
#     z = x+y+n
#     return z

# class Test:
#     def __init__(self):
#         self.x = 5
#         self.y = 5
    
#     def func(self, w):
#         m = fun(self.x,self.y,w)
#         return m

#     def testet(self):
#         with Pool(5) as p:
#             print(p.map(self.func,[1,2,3]))

# if __name__ == '__main__':    
#     asd = Test()
#     asd.testet()

#(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
#ds = tfds.load('emnist', split='train', shuffle_files=True)
#train = pd.read_csv('../emnist/emnist-bymerge-train.csv', header=None)
#test = pd.read_csv('../archive/emnist-balanced-test.csv', header=None)

#x_train = train.iloc[:, 1:]
#y_train = train.iloc[:, 0]
#x_test = test.iloc[:, 1:]
#y_test = test.iloc[:, 0]

#x_train = x_train.values
#y_train = y_train.values
#x_test = x_test.values
#y_test = y_test.values
#del train, test

#print(x_train[0])

#def rotate(image):
#    image = image.reshape([28, 28])
#    image = np.fliplr(image)
#    image = np.rot90(image)
#    return image.reshape([28 * 28])
#x_train = np.apply_along_axis(rotate, 1, x_train)
#x_test = np.apply_along_axis(rotate, 1, x_test)

#x_train = x_train.reshape(len(x_train), 28, 28)
#x_test = x_test.reshape(len(x_test), 28, 28)
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#y_test = np_utils.to_categorical(y_test, 47)
#y_train = np_utils.to_categorical(y_train, 47)

# print(tf.version.VERSION)

# x_test, y_test = get_test_val_ds("emnist")
# x_test = x_test[int(len(x_test)/2):int(len(x_test)-1)]
# y_test = y_test[int(len(y_test)/2):int(len(y_test)-1)]
# print(x_test[0])
# print(y_test[0])
#model = create_model("emnist")
#model.summary()
#model.fit(x_train, y_train, epochs=1, batch_size=128, validation_split=0.1)

#model.evaluate(x_test,y_test)

#x_train, y_train = get_train_ds(10, 1)
#x_test, y_test = get_test_val_ds()

#model = create_model_ascent()
#model.fit(x_test, y_test, epochs=1, batch_size=128, validation_split=0.1)
#model.evaluate(x_test,y_test)

#asd = model.get_weights()
#asd2 = model.get_weights()
#print(asd[1])
#r = []
#for i in range(len(asd)):
#    r.append(np.subtract(asd[i],asd2[i]))
#print(r[1])

#x = np.array([0.7336,0.7476,0.7280,0.7466,0.7243,0.7339,0.7435,0.7560, 0.7336, 0.7479])
#print(np.mean(x))
#print(np.std(x))
"""
#<class 'tensorflow.python.data.ops.dataset_ops.ShardDataset'>
#<class 'tuple'>
#<class 'tensorflow.python.framework.ops.EagerTensor'>

x = list(x_train.as_numpy_iterator())
features = []
labels = []
for i in range(len(x)):
    for j in range(len(x[i][0])):
        features.append(x[i][0][j])
        labels.append(x[i][1][j])
features = np.array(features)
labels = np.array(labels)
#x = tf.data.Dataset.from_tensor_slices(features,labels)
#x = x.batch(32)

random.seed(0)
for i in range(50):
    x = [0.0 for j in range(10)]
    x[random.randint(0,9)] = 1.0
    labels[random.randint(0,len(labels)-1)] = x

model = Sequential()

model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
"""
