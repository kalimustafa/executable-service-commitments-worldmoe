from vendor.r2_market import R2MarketHead

class R2Shallow(R2MarketHead):
    """A1: unmodified formal summary/expert-aware shallow fusion."""
    def __init__(self, config, **scales):
        assert config.use_world
        super().__init__(config, **scales)

