class ModeConfig:
    MODE_0_WEIGHT = 1.0      # Weight for standard translation (no additional weight on entities)
    MODE_1_WEIGHT = 2.0      # Weight factor for entity-aware translation
    MODE_2_WEIGHT = 0.5      # Weight factor for placeholder-based NE handling
    MODE_3_NER_WEIGHT = 0.5  # Weight for auxiliary NER loss in multi-task learning mode
