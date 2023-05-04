import torch
from vision_transfomer import *
from transformer import TransformerRationalePredictor
from utils import *


class PatchEncoder(nn.Module):
    def __init__(self, args, kernel_size, stride, num_layers, num_channels):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        patch_size = args.patch_size
        output_size = args.d_model
        num_c = num_channels
        self.layers.append(nn.Conv2d(3, num_channels, kernel_size, stride=stride))
        cnn_output_size = self.calc_output_size(patch_size, kernel_size, stride)
        for i in range(self.num_layers - 1):
            self.layers.append(nn.Sequential(nn.Conv2d(num_c, num_c, kernel_size, stride=stride)))
            cnn_output_size = self.calc_output_size(cnn_output_size, kernel_size, stride)

        self.fc = nn.Linear(cnn_output_size * cnn_output_size * num_c, output_size)

    def calc_output_size(self, input_size, kernel_size, stride, padding=0):
        output_size = ((input_size - kernel_size + 2 * padding) / stride) + 1
        return int(math.floor(output_size))

    def forward(self, X):
        '''
        inputs:
            X: [batch_size, num_channels, patch_height, patch_width]
        outputs:
            temp: [batch_size, patch_embedding_size]
        '''
        temp = X
        for i in range(self.num_layers - 1):
            temp = self.layers[i](temp)
            temp = F.relu(temp)
        temp = self.layers[-1](temp)
        temp = rearrange(temp, 'b c w h -> b (c w h)')
        temp = self.fc(F.relu(temp))
        return temp


class InvRat(torch.nn.Module):
    """
    A Transformer-based invariant rationalization model.
    """
    def __init__(self, args):
        super(InvRat, self).__init__()

        
        self.vit_encoder = get_vit_model(args.model_type, args.img_size)
        
        # initialize the rationale generator
        self.generator = TransformerRationalePredictor(num_layers=args.num_layers, d_model=args.d_model, num_heads=args.num_heads, dff=args.dff)

        # generator output layer (binary selection)
        self.generator_fc1 = torch.nn.Sequential(torch.nn.Linear(args.d_model, args.dim_generator_fc), torch.nn.ReLU())
        self.generator_fc2 = torch.nn.Linear(args.dim_generator_fc, 2)
        self.classifier_head = torch.nn.Sequential(torch.nn.Linear(args.d_model, args.dim_classifier_head), torch.nn.ReLU(), torch.nn.Linear(args.dim_classifier_head, args.num_classes))

        self.patch_encoder = PatchEncoder(args, 3, 1, 2, args.d_model)

        self.k = args.k
        self.image_size = args.img_size
        self.patch_size = args.patch_size
        self.patch_embedding_size = args.d_model

    def _get_patches(self, inputs):
        x = get_patches(self.vit_encoder, inputs)
        return x
    
    def _get_patches2(self, inputs):
        '''
        [batch, c, h, w] -> [batch, num_patches, patch_dim]
        '''
        temp = inputs
        num_patches = self.image_size // self.patch_size
        temp = rearrange(temp, 'b c (h1 h2) (w1 w2) -> b h1 w1 (c h2 w2)', w2=self.patch_size,
                         h2=self.patch_size)  ### w2 and h2 are patch_size
        temp = rearrange(temp, 'b n_w n_h (c h2 w2) -> (b n_w n_h) c h2 w2', h2=self.patch_size, w2=self.patch_size, c=3)
        temp = self.patch_encoder(temp)
        temp = rearrange(temp, '(b nw nh) d -> b (nw nh) d', b=inputs.shape[0], nw=num_patches, nh=num_patches)
        pe = positionalencoding2d(self.patch_embedding_size, num_patches, num_patches)
        temp = temp + pe.to(device)

        return temp
    
    def freeze_vit(self):
        for param in self.vit_encoder.parameters():
            param.requires_grad = False
        
    def unfreeze_vit(self):
        for param in self.vit_encoder.parameters():
            param.requires_grad = True
    
    def freeze_classifier_head(self):
        for param in self.classifier_head.parameters():
            param.requires_grad = False

    def _independent_straight_through_sampling(self, rationale_logits, mode = 'top-k'):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = torch.nn.functional.softmax(rationale_logits, -1)

        #top-K
        if mode=='top-k':
            indices = torch.topk(z[:,:,1], k=self.k).indices

            mask = torch.zeros([z.shape[0],z.shape[1]]).to(device) 
            mask.scatter_(1, indices, 1.)

            with torch.no_grad():
                neg = mask-z[:,:,1]
            
            ret = neg + z[:,:,1]

        if mode=='sigmoid':
            ret = torch.nn.functional.sigmoid(torch.squeeze(z[:,:,1]))

        if mode == 'flexible':
            mask = (z[:,:,1] == torch.max(z[:,:,1], dim=-1, keepdim=True)[0]).type(torch.IntTensor).to(device)
            with torch.no_grad():
                neg = mask-z[:,:,1]
            
            ret = neg + z[:,:,1]


        return ret

    def forward(self, inputs):
        """
        Inputs:
            inputs -- (batch_size, num-channels, width, height)
            outputs -- (batch_size, num_classes)
        """

        patches = self._get_patches2(inputs)

        ############## generator ##############
        gen_embeddings = patches
        gen_outputs = self.generator(gen_embeddings)
        gen_logits = self.generator_fc2(self.generator_fc1(gen_outputs))

        rationale = self._independent_straight_through_sampling(gen_logits)

        ############## predictor ##############
        rationale_ext = torch.unsqueeze(rationale, dim=-1)
        masked = prepare_tokens(self.vit_encoder, patches*rationale_ext, add_cls=False)
        logits = self.classifier_head(masked[:, 0, :])

        return rationale, logits
    
    def generator_params(self):
      return [p for p in self.generator.parameters()]+[p for p in self.generator_fc1.parameters()]+[p for p in self.generator_fc2.parameters()]

    def predictor_params(self):
        return [p for p in self.vit_encoder.parameters()] + [p for p in self.classifier_head.parameters()]