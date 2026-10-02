from vendor.r2_market import R2MarketHead, R2Config

class R2Base(R2MarketHead):
    """A0: exactly the existing no-world architecture, including decoder."""
    def __init__(self, config, **scales):
        assert not config.use_world
        super().__init__(config, **scales)

