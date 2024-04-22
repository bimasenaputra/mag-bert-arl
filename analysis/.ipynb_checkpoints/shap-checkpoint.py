import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import pickle
import numpy as np

def squeezed_attr(attr):
    return attr.sum(dim=2).squeeze(0)

with open("shap.pkl", "rb") as handle:
    shap = pickle.load(handle)
    
attribution, segment, text = shap

# Visualize text attributions
def show_text_attr(attrs, text):
    rgb = lambda x: '255,0,0' if x < 0 else '0,255,0'
    alpha = lambda x: abs(x) ** 0.5
    token_marks = [
        f'<mark style="background-color:rgba({rgb(attr)},{alpha(attr)})">{token}</mark>'
        for token, attr in zip(text, attrs.tolist())
    ]
    
    with open('text_attr.html', 'w') as file:
        file.write('<!DOCTYPE html>\n<html>\n<head>\n<title>Token Visualization</title>\n</head>\n<body>\n')
        file.write('<p>' + ' '.join(token_marks) + '</p>\n')
        file.write('</body>\n</html>')

show_text_attr(squeezed_attr(attribution[0]), text)

fig, (axv, axc) = plt.subplots(1,2)

# Visualize visual attributions
visual_attr = attribution[1].squeeze(0).detach().cpu().numpy()
vis_norm = clrs.LogNorm(vmin=np.min(visual_attr), vmax=np.max(visual_attr))
axv.imshow(visual_attr, cmap="magma_r", norm=vis_norm)
axv.set_title('Visual Attribution')
cbar1 = plt.colorbar(img, ax=axv) 
cbar1.set_label('Attribution Values')
axv.set_xlabel('i-th Visual Feature') 
axv.set_ylabel('Token Position')

# Visualize acoustic attributions
acoustic_attr = attribution[2].squeeze(0).detach().cpu().numpy()
ac_norm = clrs.LogNorm(vmin=np.min(acoustic_attr), vmax=np.max(acoustic_attr))
axc.imshow(acoustic_attr, cmap="magma_r", norm=ac_norm)
axc.set_title('Acoustic Attribution')
cbar2 = plt.colorbar(img, ax=axc) 
cbar2.set_label('Attribution Values')
axc.set_xlabel('i-th Acoustic Feature') 
axc.set_ylabel('Token Position')

plt.savefig('non_text_attr.png')
