require 'torch'
require 'rnn'
require 'optim'
require 'nngraph'

------ for double state --------------------------------------
local utils = require 'tools.utils'
require 'tools.DataLoader_multi'

local net_utils = require 'tools.net_utils'
-- ====================================================================================================
-- use command line options for model and training configuration
-- I may not be using some of these options in this example
-- ====================================================================================================
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a frame-level encoder-decoder sequence model for video captioning')
cmd:text()
cmd:text('Options')
-- Data
-- Data input and output settings
cmd:option('-input_h5_global','data/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_h5_object','data/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_h5_relation','data/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_h5_region','data/data.h5','path to the h5file containing the preprocessed dataset')

cmd:option('-input_json','data/data.json','path to the json file containing additional info and vocab')
cmd:option('-topic_file','','topic file')
cmd:option('-label_file', '', 'path to the file for labels')
cmd:option('-add_supervision', 0, 'whether adding extra supervsion info, 1 for true, 0 for false, else for no attention')
cmd:option('-flag','lrtest','path to the json file for vis some results')
-- model params
cmd:option('-hiddensize', 512, 'size of LSTM internal state')


-- optimization
cmd:option('-learningRate',2e-4,'learning rate')
cmd:option('-learning_rate_decay', 5e-5 ,'learning rate decay')
cmd:option('-learning_rate_decay_after',-1,'in number of epochs, when to start decaying the learning rate')
cmd:option('-learning_rate_decay_every', 5, 'every how many epochs thereafter to drop LR by half?')
cmd:option('-dropout',0.8,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-grad_clip',5,'clip gradients at this value, pass 0 to disable')

-- iteration configure
cmd:option('-batchsize', 64,'number of sequences to train on in parallel')
cmd:option('-garbage', 100, 'iter for storage garbage collection, correlating with batchsize, every garbage*batchsize samples')
cmd:option('-max_epochs',500,'number of full passes through the training data')
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')

-- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '/path/to/the/model', 'initialize network parameters from checkpoint at this path')

-- Evaluation/Checkpointing
cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-checkpoint_dir', 'checkpoints/', 'output directory where checkpoints get written')
cmd:option('-savefile','model','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-language_eval',1 , 'do language eval, computer blue and other metrics, 0 = no computer')
cmd:option('-eval_every', 1500 , 'do language eval, computer blue and other metrics, 0 = no computer')

-- SYSTEM SETTING GPU/CPU
cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-gpuid',1,'which gpu to use. -1 = use CPU')
cmd:option('-isShuffle', true, 'shuffle the train data for every epoch')
cmd:text()

-- ====================================================================================================
-- parse input params
opt = cmd:parse(arg)

opt.input_h5_global = 'features/msvd_all_sample30_frame_googlenet_bn_pool5.h5'
opt.input_h5_object = 'features/msvd_all_sample30_frame_msdn_obj_top10.h5'
opt.input_h5_relation = 'features/msvd_all_sample30_frame_msdn_relation_scoreweight_top50.h5'
opt.input_h5_region = 'features/msvd_all_sample30_frame_msdn_region_top5.h5'


opt.label_file = 'data/labels_th30.npy'
opt.input_json = 'data/info_th30.json'
opt.add_supervision = 0

opt.hiddensize = 512
opt.learningRate = 2e-4
opt.learning_rate_decay_every = 20
opt.dropout = 0.5
opt.grad_clip = 10

opt.batchsize = 64

modelpath_object = 'checkpoints_th30_googlenet_bn_sample30_msdn_obj_top10/180403-2322_double_state_attention_just_test_th30_googlenet_bn_sample30_msdn_obj_top10_26.t7'
modelpath_relation = 'checkpoints_th30_googlenet_bn_sample30_msdn_relation_scoreweight_top50/180404-1030_double_state_attention_just_test_th30_googlenet_bn_sample30_msdn_relation_scoreweight_top50_32.t7'
modelpath_region = 'checkpoints_th30_googlenet_bn_sample30_msdn_region_top5/180404-1029_double_state_attention_just_test_th30_googlenet_bn_sample30_msdn_region_top5_13.t7'


model_epoch = {modelpath_object,modelpath_relation,modelpath_region}
--model_epoch = {modelpath_object,modelpath_relation}

opt.eval_every = 100
opt.seed = 123

opt.gpuid = 2

torch.manualSeed(opt.seed)
-- ====================================================================================================
-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        require 'cudnn'
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end
---------- load model -----------------------------
print(string.format('total %d models ... ', #model_epoch))
model_num = #model_epoch
checkpoints = {}
for i=1,#model_epoch do
    local model_path = model_epoch[i]
    print('loading model from:', model_path)
    checkpoints[i] = torch.load(model_path)
end

-- ====================================================================================================
-- -- load data
-- ====================================================================================================
local loadopt = {}
loadopt.h5_file_global = opt.input_h5_global
loadopt.h5_file_object = opt.input_h5_object
loadopt.h5_file_relation = opt.input_h5_relation
loadopt.h5_file_region = opt.input_h5_region
loadopt.json_file = opt.input_json
loadopt.label_file = opt.label_file
loadopt.topic_file = opt.topic_file
local loader = DataLoader(loadopt)

opt.featdim_global = loader.featdim_global
opt.video_seq_length = loader.video_seq_length
opt.topic_size = loader.topic_size
opt.sentence_length = loader.sent_seq_length


-- =======================================================================
-- load corresponding model
-- =======================================================================
if opt.add_supervision == 0 then
  require 'tools.Seq2Seq_double_state_attention'
elseif opt.add_supervision ==1 then
  require 'tools.Seq2Seq_double_state_attention_supervision'
else
  require 'tools.Seq2Seq_double_state'
end


-- ====================================================================================================
---show some parameters
opt.vocabsize = loader:getVocabSize()
print('vocab size:', opt.vocabsize)
print('learningRate:',opt.learningRate,'sentence_length:',opt.sentence_length)
print('video_seq_length',opt.video_seq_length)
print('batchsize:',opt.batchsize)

-- ====================================================================================================
local date_info = os.date('%y%m%d-%H%M')
print('Current time info: ',date_info)
-- some thing for vis --------------------------
--this is used for pastalog to visualization----
------------------------------------------------
opt.modelName = string.format('%s',opt.flag)
opt.vis_info = string.format('h-%s', opt.hiddensize)
opt.metric = {}
opt.trainloss = {}
opt.valloss = {}

-- ====================================================================================================
-- Build the model
-- ====================================================================================================

local protos_list = {}
-- check if need to load from previous experiment

local lmopt_object = {}
lmopt_object.vocabsize = opt.vocabsize
lmopt_object.topic_size = opt.topic_size
lmopt_object.sentence_length = opt.sentence_length
lmopt_object.hiddensize = opt.hiddensize

lmopt_object.feat_seq_length = opt.video_seq_length
lmopt_object.featsize = opt.featdim_global

lmopt_object.att_length = loader.localnum_object
lmopt_object.att_feat_size = loader.featdim_object

protos_list[1] = {}
protos_list[1].lm = nn.Seq2Seq(lmopt_object)
protos_list[1].lm:importModel(checkpoints[1].lm)

-- build relation model
local lmopt_relation = {}
lmopt_relation.vocabsize = opt.vocabsize
lmopt_relation.topic_size = opt.topic_size
lmopt_relation.sentence_length = opt.sentence_length
lmopt_relation.hiddensize = opt.hiddensize

lmopt_relation.feat_seq_length = opt.video_seq_length
lmopt_relation.featsize = opt.featdim_global

lmopt_relation.att_length = loader.localnum_relation
lmopt_relation.att_feat_size = loader.featdim_relation

protos_list[2] = {}
protos_list[2].lm = nn.Seq2Seq(lmopt_relation)
protos_list[2].lm:importModel(checkpoints[2].lm)

----[[
-- build region model
local lmopt_region = {}
lmopt_region.vocabsize = opt.vocabsize
lmopt_region.topic_size = opt.topic_size
lmopt_region.sentence_length = opt.sentence_length
lmopt_region.hiddensize = opt.hiddensize

lmopt_region.feat_seq_length = opt.video_seq_length
lmopt_region.featsize = opt.featdim_global

lmopt_region.att_length = loader.localnum_region
lmopt_region.att_feat_size = loader.featdim_region

protos_list[3] = {}
protos_list[3].lm = nn.Seq2Seq(lmopt_region)
protos_list[3].lm:importModel(checkpoints[3].lm)
--]]

-- ====================================================================================================
-- run on gpu if possible
-- set all in the model protos to cuda fashion
-- ====================================================================================================
if opt.gpuid >=0 then
  for i=1,model_num do
    for k,v in pairs(protos_list[i]) do v:cuda() end
  end
end
-- ====================================================================================================
-- capture all parameters in a single 1-D array
params, grad_params = protos_list[1].lm:getParameters()
print('total number of parameters in LM: ', params:nElement())
assert(params:nElement() == grad_params:nElement())


-------------------------------------------------------------------------------------------------------
----- clone of sequence module ------------------------------------------------------------------------
for i=1,model_num do
  protos_list[i].lm:createClones()
end 

local weights_beam = {}
for i=1,model_num do
  weights_beam[i] = 1.0
end
function sample(input, opt_s)

  local context = input[1]
  local local_feats = {input[2], input[3], input[4]}
  local batch_size = context:size(2)
  local att_seq = context:clone():transpose(1,2):contiguous()
  
  local enc_ct_list = {}
  local enc_ht_list = {}
  local local_enc_ct_list = {}
  local local_enc_ht_list = {}
  local dec_ct_list = {}
  local dec_ht_list = {}

  for i=1,model_num do
    enc_ct_list[i] = torch.CudaTensor(batch_size, protos_list[i].lm.hiddensize):zero()
    enc_ht_list[i] = torch.CudaTensor(batch_size, protos_list[i].lm.hiddensize):zero()
    local_enc_ct_list[i] = torch.CudaTensor(batch_size, protos_list[i].lm.hiddensize):zero()
    local_enc_ht_list[i] = torch.CudaTensor(batch_size, protos_list[i].lm.hiddensize):zero()
    
    for t=1, protos_list[i].lm.feat_seq_length do
      local enc_out = protos_list[i].lm.encoder:forward{context[t],local_feats[i][t],enc_ct_list[i],enc_ht_list[i],local_enc_ct_list[i],local_enc_ht_list[i]}
      enc_ct_list[i],enc_ht_list[i],local_enc_ct_list[i],local_enc_ht_list[i]= unpack(enc_out)
    end
    dec_ct_list[i] = protos_list[i].lm.connectTable_c:forward{enc_ct_list[i], local_enc_ct_list[i]}
    dec_ht_list[i] = protos_list[i].lm.connectTable_h:forward{enc_ht_list[i], local_enc_ht_list[i]}
  end
  
  if opt_s.beam_size ~= nil then return beam_search({{dec_ct_list,dec_ht_list},att_seq}, {beam_size = opt_s.beam_size}) end
  
end

function beam_search(input, opt_bs)
  --print('doing beam serch with beam size ', opt_bs.beam_size)
  local batch_size = input[1][1][1]:size(1)
  --print(string.format('batch_size: %d', batch_size))
  local beam_size = opt_bs.beam_size
  local att_seq = input[2] -- for temporal attention
  local feat_dim = att_seq:size(3)
  local function compare(a,b) return a.p > b.p end -- used downstream
  --print(string.format('att_seq: %dx%dx%d', att_seq:size(1), att_seq:size(2), att_seq:size(3)))

  local seq = torch.LongTensor(opt.sentence_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(opt.sentence_length, batch_size):zero()

  for k=1, batch_size do

    local state_c_list = {}
    local state_h_list = {}
    for i=1,model_num do
      state_c_list[i] = input[1][1][i]:narrow(1,k,1):expand(beam_size, protos_list[i].lm.hiddensize_dec)
      state_h_list[i] = input[1][2][i]:narrow(1,k,1):expand(beam_size, protos_list[i].lm.hiddensize_dec)
    end
    -- we will write output predictions into tensor seq
    local beam_seq = torch.LongTensor(opt.sentence_length, beam_size):zero()
    local beam_seq_logprobs = torch.FloatTensor(opt.sentence_length, beam_size):zero()
    local beam_logprobs_sum = torch.zeros(beam_size) -- running sum of logprobs for each beam
    local logprobs_list = {} -- logprobs predicted in last time step, shape (beam_size, vocab_size+1)

    local done_beams = {}
    local att_seq_temp = att_seq:narrow(1,k,1):expand(beam_size, protos_list[1].lm.feat_seq_length, feat_dim):contiguous() 
    for t=1,opt.sentence_length+1 do
      local embed_out_list = {}
      local it, sampleLogprobs
      local new_state_c_list = {}
      local new_state_h_list = {}
      if t==1 then
        it = torch.LongTensor(beam_size):fill(opt.vocabsize+1)
        for i=1,model_num do
          embed_out_list[i] = protos_list[i].lm.embed:forward(it)
        end
        --print(it)
        --print(string.format('batch k=%d, t=%d, embed_out: %dx%d', k, t, embed_out:size(1), embed_out:size(2)))
        --print(embed_out)
        
      else
        --[[
        local logprobsf = logprobs_list[1]:float()
        for i=2,model_num do
          --local flag_a = torch.gt(logprobsf, logprobs_list[i]:float()):float()
          --local flag_b = 1 - flag_a
          --logprobsf = torch.cmul(logprobsf, flag_a) + torch.cmul(logprobs_list[i]:float(), flag_b)
          logprobsf:add(logprobs_list[i]:float())
        end
        logprobsf:div(model_num)
        ]]
        local logprobsf = weights_beam[1]*logprobs_list[1] + weights_beam[2]*logprobs_list[2] + weights_beam[3]*logprobs_list[3]
        ys, ix = torch.sort(logprobsf,2, true)

        local candidates = {}
        local cols = beam_size
        local rows = beam_size
        if t==2 then rows = 1 end  -- when t=2, we only see a beam sample for all samples are the same. pnly a row
        for c=1, cols do 
          for q=1, rows do
            local local_logprob = ys[{q,c}]
            local candidate_logprob = beam_logprobs_sum[q] + local_logprob
            table.insert(candidates,{c=ix[{q,c}], q=q, p=candidate_logprob, r=local_logprob})
          end
        end
        table.sort(candidates, compare)

        -- construct new beams
        for i=1,model_num do
          new_state_c_list[i] = state_c_list[i]:clone()
          new_state_h_list[i] = state_h_list[i]:clone()
        end
        local beam_seq_prev, beam_seq_logprobs_prev
        if t>2 then
          -- well need these as reference when we fork beams around
          beam_seq_prev = beam_seq[{ {1,t-2}, {} }]:clone()   -- when t=3, we begion to record words and logprobs
          beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-2}, {} }]:clone()
        end
        for vix=1, beam_size do
          local v = candidates[vix]
          if t > 2 then
            beam_seq[{ {1,t-2}, vix }] = beam_seq_prev[{ {}, v.q }]
            beam_seq_logprobs[{ {1,t-2}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
          end

          -- arrange states
          for i=1,model_num do
            new_state_c_list[i][vix] = state_c_list[i][v.q]
            new_state_h_list[i][vix] = state_h_list[i][v.q]
          end
          beam_seq[{ t-1, vix }] = v.c -- c'th word is the continuation
          beam_seq_logprobs[{ t-1, vix }] = v.r -- the raw logprob here
          beam_logprobs_sum[vix] = v.p -- the new (sum) logprob along this beam
          if v.c == opt.vocabsize+1 or t == opt.sentence_length+1 then
            table.insert(done_beams, {seq = beam_seq[{ {}, vix }]:clone(), 
                                      logps = beam_seq_logprobs[{ {}, vix }]:clone(),
                                      p = beam_logprobs_sum[vix]
                                     })
          end
        end

        it = beam_seq[t-1]
        for i=1,model_num do
          embed_out_list[i] = protos_list[i].lm.embed:forward(it)
        end
      end
      
      for i=1,model_num do
        if new_state_c_list[i] ~= nil then state_c_list[i], state_h_list[i] = new_state_c_list[i]:clone(), new_state_h_list[i]:clone() end
        ----------- we get the best beam_size words -------
        local dec_inputs = {embed_out_list[i], att_seq_temp, state_c_list[i], state_h_list[i]}
        local dec_out = protos_list[i].lm.decoder:forward(dec_inputs)
        state_c_list[i], state_h_list[i], logprobs_list[i] = unpack(dec_out)
      end
      --print(string.format('batch k=%d, t=%d, logprobs:%dx%d', k, t, logprobs:size(1), logprobs:size(2)))
      --print(logprobs[{{},{1,10}}])
    end

    table.sort(done_beams, compare)
    seq[{{},k}] = done_beams[1].seq
    seqLogprobs[{{},k}] = done_beams[1].logps
  end
  return seq, seqLogprobs
end





function evalLoss(split_index)

   loader:resetIterator(split_index)
   -- set evaluation mode
   for i=1,model_num do
     protos_list[i].lm:evaluate()
   end
   local n = 0
   local predictions = {}
   local vocab = loader:getVocab()
   sumError = 0
   local opt_sample = {beam_size=5}
   print('doing beam search with beam_size:', opt_sample.beam_size)
   while true do
      local data = loader:getBatch{batchsize = opt.batchsize, split = split_index}
      local feats_global, feats_object, feats_relation, feats_region, labels = 
                data.feats_global, data.feats_object, data.feats_relation, data.feats_region, data.labels
      n = n + data.feats_global:size(2)  -- feats_global: segLen * bs * featdim
      if opt.gpuid >= 0 then
        feats_global = feats_global:float():cuda()
        feats_object = feats_object:float():cuda()
        feats_relation = feats_relation:float():cuda()
        feats_region = feats_region:float():cuda()
        labels = labels:float():cuda()
      end
      
      -- -- forward the model to generate samples for each image
      
      local seq, seqLogprobs,alpha = sample({feats_global,feats_object, feats_relation, feats_region}, opt_sample)
      local sents = net_utils.decode_sequence(vocab, seq)  -- the last word is the end for fictitious sentence

      for k=1,#sents do
        local entry = {image_id = data.infos[k].id, caption = sents[k]}
        table.insert(predictions, entry)
        print(string.format('image %s: %s', entry.image_id, entry.caption))
        if #predictions % 100 == 0 or #predictions == 670 then
          print(string.format('%d/670 sentences have been generated.', #predictions))
        end
        if #predictions == 670 then
          break
        end
      end
       
      if n % 50 == 0 then collectgarbage() end
      if data.bounds.wrapped then 
        --debugger.enter()
        break
      end -- the split ran out of data, lets break out
   end
   print(string.format('total predictions are %d', #predictions))
   -- language 
   local lang_stats
   if opt.language_eval == 1 then
      local id = 'test1' 
      lang_stats = net_utils.language_eval(predictions, id)
   end

   -- set training mode
   for i=1,model_num do
     protos_list[i].lm:training()
   end

   -- return avg validation loss
   return lang_stats
end

-- the weights should be adjusted
weights_beam[1] = 0.4
weights_beam[2] = 0.4 
weights_beam[3] = 1.0
print('weights_beam: ', weights_beam[1], weights_beam[2], weights_beam[3])
lang_stats = evalLoss('test')

--[[
local lang_stats
weights_beam[1] = 1.0
while weights_beam[1] >= 0 do
  weights_beam[2] = 1.0
  while weights_beam[2] >= 0 do
    weights_beam[3] = 1.0
    while weights_beam[3] >= 0 do
      print('weights_beam: ', weights_beam[1], weights_beam[2], weights_beam[3])
      lang_stats = evalLoss('test')
      weights_beam[3] = weights_beam[3] - 0.2
    end
    weights_beam[2] = weights_beam[2] - 0.2
  end
  weights_beam[1] = weights_beam[1] - 0.2
end
]]
print ('All tasks were done........')

