local nn = require 'nn'
local cudnn = require 'cudnn'

local Convolution = nn.SpatialConvolution
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local SpatialBatchNormalization = nn.SpatialBatchNormalization

local model  = nn.Sequential()

model:add(Convolution(3, 100, 7, 7))
model:add(SpatialBatchNormalization(100, 1e-5))
model:add(ReLU())
model:add(Max(2,2))
model:add(Convolution(100, 150, 4, 4))
model:add(SpatialBatchNormalization(150, 1e-5))
model:add(ReLU())
model:add(Max(2,2))
model:add(Convolution(150, 250, 4, 4))
model:add(SpatialBatchNormalization(250, 1e-5))
model:add(ReLU())
model:add(Max(2, 2))
model:add(View(2250))
model:add(Linear(2250, 300))
model:add(ReLU())
model:add(Linear(300, 43))

cudnn.convert(model, cudnn)

print(model)
return model
