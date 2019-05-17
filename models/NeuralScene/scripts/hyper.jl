using RunTools
using NeuralScene
using NeuralScene.Run: infer, allparams

# Run
paramsamples = rand(allparams(), 10)
display.(paramsamples)
control(infer, paramsamples)