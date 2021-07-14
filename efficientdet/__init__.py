import model
from efficientnet import inject_tfkeras_modules, init_tfkeras_custom_objects

EfficientDetD0 = inject_tfkeras_modules(model.EfficientDetD0)
EfficientDetD1 = inject_tfkeras_modules(model.EfficientDetD1)
EfficientDetD2 = inject_tfkeras_modules(model.EfficientDetD2)
EfficientDetD3 = inject_tfkeras_modules(model.EfficientDetD3)
EfficientDetD4 = inject_tfkeras_modules(model.EfficientDetD4)
EfficientDetD5 = inject_tfkeras_modules(model.EfficientDetD5)
EfficientDetD6 = inject_tfkeras_modules(model.EfficientDetD6)
EfficientDetD7 = inject_tfkeras_modules(model.EfficientDetD7)
EfficientDetD7x = inject_tfkeras_modules(model.EfficientDetD7x)

init_tfkeras_custom_objects()