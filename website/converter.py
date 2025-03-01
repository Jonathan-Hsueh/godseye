from ultralytics import YOLO
model = YOLO('trainedmodel.pt')
model.export(format = "tfjs")

const [batch, features, numPredictions] = prediction.shape;
        const reshaped = prediction.reshape([numPredictions, features]);
        const data = await reshaped.array();
        
 return {
          boxes: data.map(p => p.slice(0, 4)),
          scores: data.map(p => p[4]),
          classes: data.map(p => {
            const classProbs = p.slice(5);
            
            return classProbs.indexOf(Math.max(...classProbs));
            
          })
        };        