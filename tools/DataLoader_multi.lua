------ new write 2017-09-05 Taehi ----------------------

require 'hdf5'
local utils = require 'tools.utils'
local DataLoader = torch.class('DataLoader')
local npy4th = require 'npy4th'
function DataLoader:__init(opt)
  print('DataLoader (multi) __init with opt: ')
  print(opt)
  self.isShuffle = true
  self.epoch = 1
  -- load the json file which contains additional information about the dataset
  print('DataLoader loading json file: ', opt.json_file)
  self.info = utils.read_json(opt.json_file)              -- for word and vocabulary
  self.ix_to_word = self.info.ix_to_word
  self.vocab_size = utils.count_keys(self.ix_to_word)
  --self.list = self.info.img2sent_list                     -- video VS sentence index
  self.sent2vix_map = self.info.img2sent_list             -- video VS sentence index
  --self.list_end_ix = self.info.label_end_ix               -- setence list end for each video
  self.label_end_ix = self.info.label_end_ix              -- setence list end for each video
  
  print('vocab size is ' .. self.vocab_size)

  -------- load global features  ---------------
  -- open the hdf5 file
  print('DataLoader loading h5 file for global: ', opt.h5_file_global)
  self.fh5_global = hdf5.open(opt.h5_file_global, 'r')
  -- extract image size from dataset
  local images_size_global = self.fh5_global:read('/images'):dataspaceSize()
  assert(#images_size_global == 3, '/images of global should be a 3D tensor')
  self.num_videos = images_size_global[1]
  self.video_seq_length= images_size_global[2]
  self.featdim_global = images_size_global[3]
  print(string.format('global features: read %d videos of size %dx%d', self.num_videos, self.video_seq_length, self.featdim_global))

  -------- load object features ---------------
  -- open the hdf5 file
  print('DataLoader loading h5 file for object: ', opt.h5_file_object)
  self.fh5_object = hdf5.open(opt.h5_file_object, 'r')
  local images_size_object = self.fh5_object:read('/images'):dataspaceSize()   -- 1970*10*1024*10
  assert(#images_size_object == 4, '/images of object should be a 4D tensor')
  assert(images_size_object[1] == self.num_videos)
  self.object_seq_length = images_size_object[2]
  self.featdim_object = images_size_object[3]
  self.localnum_object = images_size_object[4]
  print(string.format('read object features of size %dx%dx%d', self.object_seq_length, self.featdim_object, self.localnum_object))

  -------- load relation features ---------------
  -- open the hdf5 file
  print('DataLoader loading h5 file for relation: ', opt.h5_file_relation)
  self.fh5_relation = hdf5.open(opt.h5_file_relation, 'r')
  local images_size_relation = self.fh5_relation:read('/images'):dataspaceSize()   -- 1970*10*1024*10
  assert(#images_size_relation == 4, '/images of relation should be a 4D tensor')
  assert(images_size_relation[1] == self.num_videos)
  self.relation_seq_length = images_size_relation[2]
  self.featdim_relation = images_size_relation[3]
  self.localnum_relation = images_size_relation[4]
  print(string.format('read relation features of size %dx%dx%d', self.relation_seq_length, self.featdim_relation, self.localnum_relation))

  -------- load region features ---------------
  -- open the hdf5 file
  print('DataLoader loading h5 file for region: ', opt.h5_file_region)
  self.fh5_region = hdf5.open(opt.h5_file_region, 'r')
  local images_size_region = self.fh5_region:read('/images'):dataspaceSize()   -- 1970*10*1024*10
  assert(#images_size_region == 4, '/images of region should be a 4D tensor')
  assert(images_size_region[1] == self.num_videos)
  self.region_seq_length = images_size_region[2]
  self.featdim_region = images_size_region[3]
  self.localnum_region = images_size_region[4]
  print(string.format('read region features of size %dx%dx%d', self.region_seq_length, self.featdim_region, self.localnum_region))

  -- load in the sequence data
  -- load labels from npy file
  ----------- load labels ----------------------------------
  self.labels = npy4th.loadnpy(opt.label_file)

  local sent_labels_size = self.labels:size()
  print('total number of sentences is ' .. sent_labels_size[1])
  self.sent_seq_length = sent_labels_size[2]
  print('max sequence length of sentences is ' .. self.sent_seq_length)

  --local top_k_topic = 5
  --print('loading topic file'.. opt.topic_file)
  --print('get top k number '..top_k_topic)
  print('get word word_distribution....')
  --self.topics_raw = npy4th.loadnpy(opt.topic_file)
  --self.topics_raw = self.labels
  --self.topic_size = self.topics_raw:size(2)
  self.topic_size = self.vocab_size
  self.topics_pre = torch.LongTensor(self.num_videos, self.topic_size)
  self.topics = torch.LongTensor(self.num_videos, self.topic_size)
  -- from word distribution ...
  print('preparing word distribution')
  for i=1, sent_labels_size[1] do
    local vd_ix = self.sent2vix_map[tostring(i)]
    for j=1, self.sent_seq_length do
      local word_ix = self.labels[i][j]
      if word_ix ~= 0 then
     	 self.topics_pre[vd_ix][word_ix]=1
      end
    end
  end

  for i=1, self.num_videos do
    local tp_ix = 1
    for j=1, self.topic_size do
        local tp = self.topics_pre[i][j]
        if tp == 1 then
          self.topics[i][tp_ix] = j
          tp_ix = tp_ix+1
        end
    end
  end 

  print('word distribution samples')
  -- print(self.topics[1])

  self.splits_ix_video = {} -- {['train']= {1, 2, ..., 1200}, ['val']={1201, ..., 1300}, ['test']={1301, ..., 1970}}
  self.ix_iter_videos = {} -- init: {['train']=1, ['val']=1, ['test']=1}
  self.shuffle_ix_sent = {} -- all sentence ix of training data
  self.ix_iter_trainsents = 1
  for i,img in pairs(self.info.images) do
    local split = img.split
    if not self.splits_ix_video[split] then
      -- initialize new split
      self.splits_ix_video[split] = {}
      self.ix_iter_videos[split] = 1
    end
    table.insert(self.splits_ix_video[split], i)
  end

  -- for new training table shuffle  6513 for MSRVTT
  for i=1, self.label_end_ix[1200] do
    table.insert(self.shuffle_ix_sent, i)
  end
  utils.shuffle_table(self.shuffle_ix_sent)
  -- check shuffle_ix_sent table with list_end_ix and list -----
  print('train table label index is', self.label_end_ix[1200])
  assert(self.sent2vix_map[tostring(self.label_end_ix[1200])] == 1200, 'load train sentences index wrong')
  for k,v in pairs(self.splits_ix_video) do
    print(string.format('assigned %d images to split %s', #v, k))
  end

end

function DataLoader:resetIterator(split)
  self.ix_iter_videos[split] = 1
end

function DataLoader:getVocabSize()
  return self.vocab_size
end

function DataLoader:getVocab()
  return self.ix_to_word
end

function DataLoader:getSeqLength()
  return self.sent_seq_length
end

function DataLoader:getBatchAll(opt)
  local split = opt.split  -- lets require that user passes this in, for safety
  local batchsize = opt.batchsize -- how many images get returned at one time (to go through CNN)

  -- pick an index of the datapoint to load next
  --local img_batch_raw = torch.FloatTensor(batchsize , self.video_seq_length, self.featdim_global):zero()
  local global_batch_raw = torch.FloatTensor(batchsize , self.video_seq_length, self.featdim_global):zero()
  local object_batch_raw = torch.FloatTensor(batchsize, self.object_seq_length, self.featdim_object, self.localnum_object):zero()
  local relation_batch_raw = torch.FloatTensor(batchsize, self.relation_seq_length, self.featdim_relation, self.localnum_relation):zero()
  local region_batch_raw = torch.FloatTensor(batchsize, self.region_seq_length, self.featdim_region, self.localnum_region):zero()
  local label_batch = torch.LongTensor(batchsize , self.sent_seq_length):zero()
  local topic_batch = torch.LongTensor(batchsize, self.topic_size):zero()

  local wrapped = false --indicate if traverse to the end of training data and back to the index 1
  local infos = {}
  local shuffle_ix_sent = self.shuffle_ix_sent

  local max_index = #shuffle_ix_sent
  
  for i=1, batchsize do

      local ri = self.ix_iter_trainsents -- the index of current iteration
      ri_next = ri + 1
      if ri_next > max_index then
        ri_next = 1
        wrapped = true
        -- shuffle train split
        utils.shuffle_table(self.shuffle_ix_sent)
        print(split .. ' data is shuffled...')
        self.epoch = self.epoch + 1
      end -- wrap back around
      self.ix_iter_trainsents = ri_next
      --debugger.enter()
      local ix_sent = shuffle_ix_sent[ri]

      local ix_video = self.sent2vix_map[tostring(ix_sent)] 
      -- if ix_video > 1200 then print('get wrong training video index...') end
      assert(ix_video <= 1200, 'get wrong training video index...')
      -- debugger.enter()
      local feat_global = self.fh5_global:read('/images'):partial({ix_video,ix_video}, {1, self.video_seq_length}, {1, self.featdim_global})
      global_batch_raw[i] = feat_global

	    object_batch_raw[i] = self.fh5_object:read('/images'):partial({ix_video, ix_video}, 
                    {1, self.object_seq_length}, {1, self.featdim_object}, {1,self.localnum_object})
      relation_batch_raw[i] = self.fh5_relation:read('/images'):partial({ix_video, ix_video}, 
                    {1, self.relation_seq_length}, {1, self.featdim_relation}, {1,self.localnum_relation})
      region_batch_raw[i] = self.fh5_region:read('/images'):partial({ix_video, ix_video}, 
                    {1, self.region_seq_length}, {1, self.featdim_region}, {1,self.localnum_region})
      						
      label_batch[i] = self.labels[{ix_sent, {1,self.sent_seq_length}}]
      topic_batch[i] = self.topics[{ix_video, {1,self.topic_size}}]
      
      -- and record associated info as well
      local info_struct = {}
      info_struct.id = self.info.images[ix_video].id
      table.insert(infos, info_struct)
  end
  local data = {}
  data.feats_object = object_batch_raw:transpose(1,2):transpose(3,4):contiguous()      -- seqLen*bs*10*1024
  data.feats_relation = relation_batch_raw:transpose(1,2):transpose(3,4):contiguous()  -- seqLen*bs*50*1024
  data.feats_region = region_batch_raw:transpose(1,2):transpose(3,4):contiguous()      -- seqLen*bs*5 *1024
  data.topics = topic_batch:contiguous()  -- bs*vocabSize
  data.feats_global = global_batch_raw:transpose(1,2):contiguous()  -- seqLen*batchsize*feat_len  -- seqLen*bs*1024
  data.labels = label_batch:transpose(1,2):contiguous() -- note: make label sequences go down as columns -- sentLen*bs -- 12*bs
  --data.bounds = {it_pos_now = self.ix_iter_trainsents, it_max = max_index, wrapped = wrapped}
  data.bounds = {sent_ix_next = self.ix_iter_trainsents, sent_ix_max = max_index, wrapped = wrapped}
  data.infos = infos -- record video ids of current batch
  
  return data
end

--[[
  Split is a string identifier (e.g. train|val|test)
  Returns a batch of data:
  - X (N,3,H,W) containing the images
  - y (L,M) containing the captions as columns (which is better for contiguous memory during training)
  - info table of length N, containing additional information
  The data is iterated linearly in order. Iterators for any split can be reset manually with resetIterator()
--]]
function DataLoader:getBatch(opt)
  local split = opt.split -- lets require that user passes this in, for safety
  local batchsize = opt.batchsize -- how many images get returned at one time (to go through CNN)

  local ixs_video = self.splits_ix_video[split]
  assert(ixs_video, 'split ' .. split .. ' not found.')

  -- pick an index of the datapoint to load next
  local global_batch_raw = torch.FloatTensor(batchsize , self.video_seq_length, self.featdim_global):zero()
  local object_batch_raw = torch.FloatTensor(batchsize, self.object_seq_length, self.featdim_object, self.localnum_object):zero()
  local relation_batch_raw = torch.FloatTensor(batchsize, self.relation_seq_length, self.featdim_relation, self.localnum_relation):zero()
  local region_batch_raw = torch.FloatTensor(batchsize, self.region_seq_length, self.featdim_region, self.localnum_region):zero()
  local label_batch = torch.LongTensor(batchsize, self.sent_seq_length):zero()
  local topic_batch = torch.LongTensor(batchsize, self.topic_size):zero()

  local max_index = #ixs_video
  local wrapped = false
  local infos = {}
  for i=1, batchsize do

    local ri = self.ix_iter_videos[split] -- get next index from iterator
    local ri_next = ri + 1 -- increment iterator
    if ri_next > max_index then
      ri_next = 1
      wrapped = true
    end -- wrap back around
    self.ix_iter_videos[split] = ri_next
    ix_video = ixs_video[ri]
    assert(ix_video ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)

    -- fetch the image from h5
    local feat_global = self.fh5_global:read('/images'):partial({ix_video,ix_video},
                            {1, self.video_seq_length}, {1, self.featdim_global})
    global_batch_raw[i] = feat_global

    object_batch_raw[i] = self.fh5_object:read('/images'):partial({ix_video, ix_video}, 
                    {1, self.object_seq_length}, {1, self.featdim_object}, {1,self.localnum_object})
    relation_batch_raw[i] = self.fh5_relation:read('/images'):partial({ix_video, ix_video}, 
                    {1, self.relation_seq_length}, {1, self.featdim_relation}, {1,self.localnum_relation})
    region_batch_raw[i] = self.fh5_region:read('/images'):partial({ix_video, ix_video}, 
                    {1, self.region_seq_length}, {1, self.featdim_region}, {1,self.localnum_region})
    
    local info_struct = {}
    info_struct.id = self.info.images[ix_video].id
    table.insert(infos, info_struct)
  end

  local data = {}
  data.feats_object = object_batch_raw:transpose(1,2):transpose(3,4):contiguous()      -- seqLen*bs*10*1024
  data.feats_relation = relation_batch_raw:transpose(1,2):transpose(3,4):contiguous()  -- seqLen*bs*50*1024
  data.feats_region = region_batch_raw:transpose(1,2):transpose(3,4):contiguous()      -- seqLen*bs*5 *1024
  data.feats_global = global_batch_raw:transpose(1,2):contiguous()  -- seqLen*batchsize*feat_len  -- seqLen*bs*1024
  data.labels = label_batch:transpose(1,2):contiguous() -- note: make label sequences go down as columns -- sentLen*bs -- 12*bs
  data.topics = topic_batch:contiguous()  -- bs*vocabSize
  data.bounds = {video_ix_next = self.ix_iter_videos[split], video_ix_max = max_index, wrapped = wrapped}
  data.infos = infos -- record video ids of current batch

  return data
end

