from modules.netFoundConfigBase import netFoundNoPayloadConfig



class netFoundSmall(netFoundNoPayloadConfig):
    def __init__(self, **kwargs):
        super().__init__(hidden_size=512, num_hidden_layers=4, num_attention_heads=4, intermediate_size=768, **kwargs)

class netFoundBase(netFoundNoPayloadConfig):
    def __init__(self, **kwargs):
        super().__init__(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=1152, **kwargs)

class netFoundLarge(netFoundNoPayloadConfig):
    def __init__(self, **kwargs):
        super().__init__(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=2624, **kwargs)


CONFIG_SIZES = {
    "small": netFoundSmall,
    "base": netFoundBase,
    "large": netFoundLarge,
    None: netFoundBase,
}