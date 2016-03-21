require 'sys'

require 'nn'
local nets = {}
nets[#nets+1] = require 'vgg_a'

local libs = {}
libs[#libs+1] = {nn.SpatialConvolutionMM, nn.SpatialMaxPooling, nn.ReLU, 'BDHW', 'nn'}
libs[#libs+1] = {nn.SpatialConvolutionBHWD, nn.SpatialMaxPoolingBHWD, nn.ReLU, 'BHWD', 'nnBHWD'}

steps = 10 -- nb of steps in loop to average perf
nDryRuns = 10

function makeInput(config, size)
   local layout = config[4]
   local osize
   if layout == 'BDHW' then
      osize = size
   elseif layout == 'DHWB' then
      osize = {size[2],size[3],size[4],size[1]}
   elseif layout == 'BHWD' then
      osize = {size[1], size[3], size[4], size[2]}
   end
   return torch.randn(torch.LongStorage(osize))
end

for i=1,#nets do
   for j=1,#libs do
      collectgarbage()
      local model,model_name,size = nets[i](libs[j])
      local input = makeInput(libs[j],size)
      local lib_name = libs[j][5]
      print('ModelType: ' .. model_name, 'Kernels: ' .. lib_name,
            'Input shape: ' .. input:size(1) .. 'x' .. input:size(2) ..
               'x' .. input:size(3) .. 'x' .. input:size(4))

      -- dry-run
      for i=1,nDryRuns do
         model:zeroGradParameters()
         local output = model:updateOutput(input)
         local gradInput = model:updateGradInput(input, output)
         model:accGradParameters(input, output)
         collectgarbage()
      end

      local tmf, tmbi, tmbg
      sys.tic()
      for t = 1,steps do
         output = model:updateOutput(input)
      end
      tmf = sys.toc()/steps
      print(string.format("%-30s %25s %10.2f", lib_name, ':updateOutput():', tmf*1000))

      collectgarbage()
      sys.tic()
      for t = 1,steps do
         model:updateGradInput(input, output)
      end
      tmbi = sys.toc()/steps
      print(string.format("%-30s %25s %10.2f", lib_name, ':updateGradInput():', tmbi*1000))

      collectgarbage()
      sys.tic()
      local ok = 1
      for t = 1,steps do
         ok = pcall(function() model:accGradParameters(input, output) end)
      end
      tmbg = sys.toc()/steps
      if not ok then
         print(string.format("%-30s %25s %s", lib_name, ':accGradParameters():', 'FAILED!'))
      else
         print(string.format("%-30s %25s %10.2f", lib_name, ':accGradParameters():', tmbg*1000))
      end
      print(string.format("%-30s %25s %10.2f", lib_name, ':Forward:', (tmf)*1000))
      print(string.format("%-30s %25s %10.2f", lib_name, ':Backward:', (tmbi+tmbg)*1000))
      print(string.format("%-30s %25s %10.2f", lib_name, ':TOTAL:', (tmf+tmbi+tmbg)*1000))
      print()
   end
end

print('')
