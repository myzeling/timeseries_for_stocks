# Timeseries_for_stocks
本项目的主要目的是为了预测股票价格的时间序列。如何将数据喂给深度学习的神经网络是一件令人头疼的事情，因为股票数据是一系列的面板数据，每个股票-每个时间段-n多重特征这导致了我们的数据是一个四维的数据而非原本的代码也就是Nonestationary-transformer这样一个课题所处理的时间序列数据。所以，如何去训练我们的模型，这是一个十分重要的问题。

## 本项目对股票序列的处理方案
我考虑了两种方式，一种方式是训练一个笼统集中的大模型，还有一种方式是对每一个股票分别建立时间序列的预测模型。当然，或许还有第三种方式，那就是建立一个行业大模型，然后去分别预测。如何建立行业大模型是一个有意思的问题，这个问题我们留到以后再说，现在是说不了一点了。这个项目解决的问题就是每个股票分别进行建模和预测并重新组合数据获取数据的问题。

## 数据格式是一个重要的问题
为了帮助新手同学们快速上手，这里介绍我们的数据格式。为了使得上千只股票的模型能够在程序中自动训练而不必进行人为干预，我采用了字典存储的形式传入我们的数据。也就是说，如果大家的原始数据长这样：
|date|id|feature1|feature2|feature3|...|
|----|----|----|----|----|----|
|2022-1-1|1|0.1|0.1|0.1|...|
|2022-1-2|1|0.2|0.2|0.3|...|
|...|...|...|...|...|...|
|2022-1-1|1888|0.5|0.6|0.3|...|
|2023-5-28|1888|0.8|0.6|0.5|...|
也就是说，多只股票多个特征多段时间的混合数据。我们怎么处理的？对于每一个股票，我们都把它的时间序列提取出来存放到一个字典的键值对里面。其中键就是股票id，值就是该股票的时间序列。

### 如何读取你的数据
那么如何读取你的数据呢？我的数据格式是用feather格式保存的，但是你的不一定。那么请你看到文件run.py中这段代码：
```
df = feather.read_dataframe('../daily.feather')
df = df[df['date']>='2000-01-01']
df.replace([np.inf, -np.inf], 0.00001, inplace=True)
df.fillna(0, inplace=True)

df['date'] = pd.to_datetime(df['date'])
grouped = df.groupby('id')
stock_dict = {}

for stock, group in grouped:
    stock_dict[stock] = group
    del stock_dict[stock]['id']
    del stock_dict[stock]['Markettype'] #Markettype这个列是我的数据中自带存在的列，如果你的数据里面没有，也请你把这行代码删掉或者注释掉
```
请你将你需要读取的数据改为你想要的数据读取格式，比如说你的数据是保存在excel里面的，那你就用pd.read_excel(yourpath)；如果你的数据保存在csv里面，你就用pd.read_csv(yourpath)，请务必注意，我在后面添加了时间的筛选条件：我把2000年1月1日之后的数据筛选出来了，如果你不需要，请你删除。

## 运行环境问题
由于这个项目本身是在linux下训练完成的，在windows下调用服务器的时候会报错，这是令人十分头疼的一件事情，所以这个项目还是建议各位用linux来跑。本人并不具备将本项目迁移到windows上跑的条件，然后小学了一手linux，真是令人头秃啊。另外，本项目所需的环境十分简单，在requirement.txt中均可找到，如果用云端的服务器跑，基本上都配好了，不需要另外装配。新手同学直接 pip install requirements.txt即可。

## 模型和参数的问题

### model
感谢清华的刘勇老师提供的开源模型。本项目支持以下模型的训练：transformer、informer、autoformer、ns_transformer、ns_informer、ns_autoformer。模型的选择在run.py中的model参数可以选择。

### data
另外，加载数据的时候，请大家选择custom，这个是我们根据时间序列来切割的训练集、验证集和测试集。不要选择Train_Test，那个是一体化的训练模型的切割数据法。本代码中适配

### features
这里的features选择的是预测模式。S是单特征的自预测，也就是价格（收益）预测价格（收益），没有其它变量一起预测。如果是M就是多个时间序列变量一起预测；选择MS，就是多个变量参与建模，仅输出单变量结果。各位根据自己的需求选择即可

### target
这是被预测的目标变量，在我们这个代码里面是多测单的，所以我在处理数据的时候都适配的多测单的形式，这里的target就是我们最后所需要去预测的变量（列）

### freq
fraq参数是一个被我弃用的参数，因为我们的数据都是年月日，所以我适配的代码也是选择了年月日的训练格式。如果你有更精细化的时间需求，如小时、刻、分钟、秒、毫秒，那么我们的代码是会出错的。详情请你从layers中的embed的时间编码格式去更改，另外，在数据处理的dataprovider部分，你也需要重新去做自己的data_stamp，然后再选择freq参数，那么这个参数就可以发挥其本身的作用了

### checkpoints
这个是一个文件路径，是存储训练好的大模型的路径，你可以自己编辑路径。

### seq_len,label_len,pred_len
初学者会难以理解什么是seq_len、label_len和pred_len。这里做一个粗浅的解释，当seq_len=10，label_len=5,pred_len = 2的时候，我们就是观测1-10这十个数据那么我们的X变量就是1-10这十个数据。我们的变量Y就是从seq_len倒着往前数label_len个数字，也就是5个数，再加上pred_len，也就是最后是7个值。所以我们预测了7个值，但是这里面有五个值是已知道的，只有11-12这两个值是真正被预测出来的。label_len起到了一个帮助我们校验的作用

### enc_in,dec_in,c_out
这里的三个变量分别是编码维度，解码维度和输出维度。编码解码必须是同一个值，而且必须是你输入模型进行训练的值，也就是说，你输入模型的变量有多少个（包含特征数量），你就要输多少维。c_out代表的是你的预测变量的维度，如果你只有一个变量要预测，c_out就是1


### embed
这个参数跟我提到的freq还有时间的嵌入层息息相关，你可以到上面所述的地方去找到原代码所支持的时间格式，也可以自己编辑时间格式，但是如果你只想简单复现我的代码，你就选择fixed，不然你跑不通的。

### do_predict
不建议用，因为这个地方我是没有适配的，它是在一段已知时间序列的基础上去预测pred_len长度的，还未发生的时间序列。这个功能对于我做股票回测来说十分鸡肋，所以我没有适配这个接口，各位有兴趣自己去看。

### 其它参数
关于模型的隐藏层，d_model，head等参数，这是更深层的transformer领域所需涉及的东西，当然也是很基础的参数。介绍都不难看懂，各位有兴趣可以自行调整，另外学习率，gpu什么的参数，相信各位应该也不能看懂。看不懂也不要紧，按照我所说的调整数据到需要的格式，我就能保证各位能够相应跑出结果了。

## 我得到的结果在哪里
如果你指的是模型，那么它们都存在checkpoints里面，但是注意了。我训练5000个长度为1500左右的模型，我估算需要的硬盘空间大概有个300g吧。所以我采取的办法是：去模型留结果。这一步只好手动改了，因为内存满了之后，程序会被kill，需要人为重启。
```
pred = []
    for stock, data in stock_dict.items():
        if stock>=2000 and stock<=2299:
            if data['date'].min() <= pd.to_datetime('2018-10-31') and data['date'].max()>=pd.to_datetime('2021-05-31'):
                try:
                    print("现在是stock",stock)
                    exp = Exp(args,stock,data)
                    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    tempt = exp.test(setting, test=1)
                    pred.append(tempt)
                    if args.do_predict:
                        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                        exp.predict(setting, True)
                    del exp
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error message: {str(e)}")

    try:
        pred = np.concatenate(pred, axis=0)
        pred = pred.reshape(-1, pred.shape[-1])
        pred = pd.DataFrame(pred, columns=['pred', 'year', 'month', 'day','id'])
        feather.write_dataframe(pred, f'./pred/pred_2000_to_2299.feather')
    except Exception as e:
        print(f"Error message: {str(e)}")
```
正常人应该知道我把文件存在哪里了吧。建议大家存feather格式，因为这个占用内存少，csv格式占用内存特别大。feather的安装代码：
```
pip install feather-format
```
## linux机部署本项目的操作
```
git clone git@github.com:myzeling/timeseries_for_stocks.git
```
然后
```
cd timeseries_for_stocks
```
各位把该改好的参数直接改好，就能去开炮了
```
python run.py
```
注意到run里面去改test部分得到自己的文件夹。