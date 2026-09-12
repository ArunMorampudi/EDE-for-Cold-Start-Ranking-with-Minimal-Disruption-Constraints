from pathlib import Path
import subprocess
from PIL import Image,ImageOps,ImageDraw
root=Path('revision1/qa/word')
for pdf in root.glob('*.pdf'):
    folder=Path('revision1/qa/pages')/pdf.stem; folder.mkdir(parents=True,exist_ok=True)
    binary=Path('C:/Users/Arun/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe')
    subprocess.run([str(binary),'-png','-r','108',str(pdf),str(folder/'page')],check=True)
    pages=sorted(folder.glob('page-*.png'),key=lambda x:int(x.stem.split('-')[-1]))
    for offset in range(0,len(pages),6):
        sheet=Image.new('RGB',(1020,960),'#cccccc'); draw=ImageDraw.Draw(sheet)
        for j in range(offset,min(offset+6,len(pages))):
            img=Image.open(pages[j]); img.thumbnail((330,440))
            x=(j-offset)%3*340; y=(j-offset)//3*480
            sheet.paste(img,(x,y+22)); draw.text((x+5,y+4),f'{pdf.stem} p{j+1}',fill='black')
        sheet.save(folder/f'contact-{offset+1}.png')
    print(pdf.name,len(pages))
