
class FeatureExtractorDDPM(FeatureExtractor):
    '''
    Wrapper to extract features from pretrained DDPMs.
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''

    def __init__(self, steps: List[int], blocks: List[int], **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        # blocks = [5,6,7,8,12]
        # Save decoder activations
        for idx, block in enumerate(self.model.output_blocks):  # total 18 blocks
            if idx in blocks:
                # print(f'block : {block.__class__.__name__}')
                # registered feature is hooked ...
                # block.register_forward_hook(self.save_hook)
                block.register_forward_hook(save_out_hook)
                # register_forward_hook(function)

                # def save_out_hook(self, inp, out):
                # out !!
                #    save_tensors(self, out, 'activations')
                #    return out

                # out
                name = 'activations'
                # features = block.features

                module = self  # output features (hidden states)
                # print(f'block out features : {type(features)}')
                """
                def save_tensors(module: nn.Module, features, name: str):
                    # Process and save activations in the module. 
                    if type(features) in [list, tuple]:
                        features = [f.detach().float() if f is not None else None
                                    for f in features]
                        setattr(module, name, features)
                    elif isinstance(features, dict):
                        features = {k: f.detach().float() for k, f in features.items()}
                        setattr(module, name, features)
                    else:
                        setattr(module, name, features.detach().float())
                """

                # what is block.activation ?

                # print(f'Block {idx} activation : {block.activations}')
                self.feature_blocks.append(block)

    def _load_pretrained_model(self, model_path, **kwargs):
        import inspect
        import guided_diffusion.guided_diffusion.dist_util as dist_util
        from guided_diffusion.guided_diffusion.script_util import create_model_and_diffusion

        # Needed to pass only expected args to the function
        argnames = inspect.getfullargspec(create_model_and_diffusion)[0]
        expected_args = {name: kwargs[name] for name in argnames}
        self.model, self.diffusion = create_model_and_diffusion(**expected_args)
        """
        self.model.load_state_dict(
            dist_util.load_state_dict(model_path, map_location="cpu")
        )
        self.model.to(dist_util.dev())
        if kwargs['use_fp16']:
            self.model.convert_to_fp16()
        """
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, noise=None):
        activations = []
        for t in self.steps:
            t = torch.tensor([t]).to(x.device)
            noisy_x = self.diffusion.q_sample(x, t, noise=noise)
            self.model(noisy_x, self.diffusion._scale_timesteps(t))
            # Extract activations
            for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None

        # Per-layer list of activations [N, C, H, W]
        return activations
