require 'torch'
require 'optim'
require 'os'
require 'optim'
require 'xlua'
require 'cunn'
require 'cudnn'
require 'gnuplot'

cudnn.benchmark = true
cudnn.fastest = true

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)

local WIDTH, HEIGHT = 48, 48
local DATA_PATH = (opt.data ~= '' and opt.data or './data/')

local ap_flag = true

torch.setdefaulttensortype('torch.DoubleTensor')

-- torch.setnumthreads(1)
--torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

--function resize(img)
--    return image.scale(img, WIDTH,HEIGHT)
--end

function resize(img,x1,y1,x2,y2)
	img = image.crop(img, x1, y1, x2, y2)
	img = image.scale(img, WIDTH, HEIGHT)
	--img = image.rgb2yuv(img)
	img = (img - torch.mean(img))
	img = img/torch.std(img)
	return img
end


function transformInput(inp,x1,y1,x2,y2)
    return resize(inp,x1,y1,x2,y2)
end

function getTrainSample(dataset, idx)
    r = dataset[idx]
    classId, track, file = r[9], r[1], r[2]
    x1_coord, y1_coord, x2_coord, y2_coord = r[5], r[6], r[7], r[8]
    file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
    return transformInput(image.load(DATA_PATH .. '/train_images/'..file),x1_coord,y1_coord,x2_coord,y2_coord)
end

function getTrainLabel(dataset, idx)
    return torch.LongTensor{dataset[idx][9] + 1}
end

function getTestSample(dataset, idx)
    r = dataset[idx]
    x1_coord, y1_coord, x2_coord, y2_coord = r[4], r[5], r[6], r[7]
    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
    return transformInput(image.load(file),x1_coord,y1_coord,x2_coord,y2_coord)
end

function getIterator(dataset)
    return tnt.DatasetIterator{
       	dataset = tnt.BatchDataset{
            batchsize = opt.batchsize,
            dataset = dataset
	}
    }
end

local trainData = torch.load(DATA_PATH..'train.t7')
local testData = torch.load(DATA_PATH..'test.t7')

trainDataset = tnt.SplitDataset{
    partitions = {train=0.9, val=0.1},
    initialpartition = 'train',

    dataset = tnt.ShuffleDataset{
        dataset = tnt.ListDataset{
            list = torch.range(1, trainData:size(1)):long(),
            load = function(idx)
                return {
                    input =  getTrainSample(trainData, idx),
                    target = getTrainLabel(trainData, idx)
                }
            end
        }
    }
}

testDataset = tnt.ListDataset{
    list = torch.range(1, testData:size(1)):long(),
    load = function(idx)
        return {
            input = getTestSample(testData, idx),
            sampleId = torch.LongTensor{testData[idx][1]}
        }
    end
}


train_error_tbl = {}
train_loss_tbl = {}
epoch_tbl = {}
classCounts = torch.Tensor(43)
for ap_i = 1, trainData:size(1) do
    classCounts[trainData[ap_i][9]+1] = classCounts[trainData[ap_i][9]+1] + 1
end


local model = require("models/".. opt.model)
model:cuda()
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.CrossEntropyCriterion()
criterion:cuda()
local clerr = tnt.ClassErrorMeter{topk = {1}}
local timer = tnt.TimeMeter()
local batch = 1

-- print(model)

engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    if state.training then
        mode = 'Train'
    else
        mode = 'Val'
    end
end


engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    clerr:add(state.network.output, state.sample.target)
    if opt.verbose == true then
        print(string.format("%s Batch: %d/%d; avg. loss: %2.4f; avg. error: %2.4f",
                mode, batch, state.iterator.dataset:size(), meter:value(), clerr:value{k = 1}))
    else
        xlua.progress(batch, state.iterator.dataset:size())
    end
    batch = batch + 1 
    timer:incUnit()
end

local inp_gpu, tgt_gpu = torch.CudaTensor(), torch.CudaTensor()
engine.hooks.onSample = function(state)
      inp_gpu:resize(state.sample.input:size()):copy(state.sample.input)
      --print(state.sample.input:std())
      --print(state.sample.input:mean())
      state.sample.input = inp_gpu
      if ap_flag == true then
	      tgt_gpu:resize(state.sample.target:size()):copy(state.sample.target)
	      state.sample.target = tgt_gpu
      end
end

engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    mode, meter:value(), clerr:value{k = 1}, timer:value()))
    --table.insert(epoch_tbl, epoch)
    if state.training then
	table.insert(train_error_tbl, clerr:value{k = 1})
	table.insert(train_loss_tbl, meter:value())
    --else
	--val_error[epoch] = clerr:value{k = 1}
	--val_loss[epoch] = meter:value()
    end
end

local epoch = 1

while epoch <= opt.nEpochs do
    trainDataset:select('train')
    engine:train{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset),
        optimMethod = optim.sgd,
        maxepoch = 1,
        config = {
            learningRate = opt.LR,
            momentum = opt.momentum
        }
    }

    trainDataset:select('val')
    engine:test{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset)
    }
    print('Done with Epoch '..tostring(epoch))
    table.insert(epoch_tbl, epoch)
    epoch = epoch + 1
end

local submission = assert(io.open(opt.logDir .. "/submission.csv", "w"))
submission:write("Filename,ClassId\n")
batch = 1

engine.hooks.onForward = function(state)
    local fileNames  = state.sample.sampleId
    local _, pred = state.network.output:max(2)
    pred = pred - 1
    for i = 1, pred:size(1) do
        submission:write(string.format("%05d,%d\n", fileNames[i][1], pred[i][1]))
    end
    xlua.progress(batch, state.iterator.dataset:size())
    batch = batch + 1
end

engine.hooks.onEnd = function(state)
    submission:close()
end

ap_flag = false
engine:test{
    network = model,
    iterator = getIterator(testDataset)
}

model:clearState()
torch.save(opt.logDir .. "/model.t7",model)
print('Model saved')

epoch_array = torch.Tensor(epoch_tbl)
--print(epoch_array:size())
train_error = torch.Tensor(train_error_tbl)
--print(train_error:size())
train_loss = torch.Tensor(train_loss_tbl)
--print(train_loss:size())
gnuplot.pngfigure(opt.logDir .. "/train_plot.png")
gnuplot.plot({'Train_Error', epoch_array, train_error, '-'})
gnuplot.xlabel('Epochs ----->')
gnuplot.ylabel('Average Error ----->')
gnuplot.plotflush()

print("The End!")
