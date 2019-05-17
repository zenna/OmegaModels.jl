using RunTools
using NeuralScene
using NeuralScene.Run: infer, allparams

# Run
control(infer, rand(allparams(), 2))