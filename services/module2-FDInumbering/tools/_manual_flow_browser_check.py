"""Exercise drawing, canonical-box handoff and reviewed geometry on a synthetic fixture."""
import argparse
import io
import json
import re
import zipfile
from playwright.sync_api import sync_playwright, expect
from dental_fdi.paths import PROJECT_ROOT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://localhost:8529');args=parser.parse_args()
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='chrome',headless=True)
        page=browser.new_page(accept_downloads=True,viewport={'width':1440,'height':1000})
        page.goto(args.url)
        page.locator('input[type=file]:not([multiple])').wait_for(state='attached',timeout=90000)
        page.locator('input[type=file]:not([multiple])').set_input_files(str(PROJECT_ROOT/'samples/sample_opg_1.jpg'))
        page.get_by_text('Draw a tooth box on the X-ray',exact=True).click(timeout=90000)

        def select(label,value):
            widget=page.get_by_test_id('stSelectbox').filter(has_text=label)
            field=widget.locator('input')
            if field.input_value()==value:
                return
            field.click()
            if field.get_attribute('readonly') is None:
                field.fill(value)
            page.get_by_role('option',name=value,exact=True).click()

        def click_points(frame,points):
            for index,(x,y) in enumerate(points):
                frame.locator('#landmark').select_option(index=index)
                canvas=frame.locator('canvas');canvas.wait_for();expect(canvas).not_to_have_attribute('height','150')
                rect=canvas.bounding_box()
                width=int(canvas.get_attribute('width'));height=int(canvas.get_attribute('height'))
                canvas.click(position={'x':x/width*rect['width'],'y':y/height*rect['height']})
                page.wait_for_timeout(250)

        for fdi,coords in [('37',(600,200,660,400)),('38',(680,200,780,400))]:
            select('FDI code for drawn box',fdi)
            print('Drawing FDI',fdi,flush=True)
            frame=page.frame_locator('iframe[title*="tooth_box_picker"]')
            canvas=frame.locator('canvas');canvas.wait_for();expect(canvas).to_have_attribute('width','1024');canvas.scroll_into_view_if_needed()
            rect=canvas.bounding_box();width=int(canvas.get_attribute('width'));height=int(canvas.get_attribute('height'))
            x1,y1,x2,y2=coords
            page.mouse.move(rect['x']+x1/width*rect['width'],rect['y']+y1/height*rect['height'])
            page.mouse.down();page.mouse.move(rect['x']+x2/width*rect['width'],rect['y']+y2/height*rect['height'],steps=8);page.mouse.up()
            button=page.get_by_role('button',name='Save drawn box for selected FDI',exact=True)
            expect(button).to_be_enabled(timeout=30000);button.click()
            expect(page.get_by_text('Reviewed FDI boxes:',exact=False)).to_contain_text(fdi,timeout=30000)
            page.wait_for_timeout(500)
        print('Boxes saved; checking downstream panels',flush=True)
        select('Tooth for bone-level measurement','37')
        page.get_by_text('Draw green X/Y guides and a red line',exact=False).click()
        reference=page.get_by_test_id('stExpander').filter(has_text='Draw green X/Y guides and a red line').frame_locator('iframe')
        click_points(reference,[(200,250),(400,250),(400,100),(400,400),(220,160),(380,320)])
        page.get_by_text(re.compile(r'Red.X:'),exact=False).wait_for(timeout=30000)
        assert abs(float(re.findall(r'[0-9]+[.][0-9]+',page.get_by_text(re.compile(r'Red.X:'),exact=False).inner_text())[0])-45)<.5
        select('Third molar to measure','38')
        page.get_by_text('38: amber axis. 37: cyan reference axis.',exact=True).wait_for(timeout=30000)
        molar=page.locator('.st-key-molar_measurement_panel').frame_locator('iframe')
        click_points(molar,[(735,380),(685,250),(630,380),(630,220)])
        expect(page.get_by_text('Draft angle:',exact=False)).to_contain_text(re.compile(r'21[.][0-9]+'),timeout=30000)
        assert abs(float(re.findall(r'[0-9]+[.][0-9]+',page.get_by_text('Draft angle:',exact=False).inner_text())[0])-21.04)<.5
        label=page.get_by_text('I reviewed the identities of 38/37 and both root-to-crown axes',exact=True)
        label.evaluate("el=>el.scrollIntoView({block:'center'})");label.click()
        bone=page.locator('.st-key-bone_measurement_panel').frame_locator('iframe')
        canvas=bone.locator('canvas');canvas.wait_for();expect(canvas).not_to_have_attribute('height','150')
        width=int(canvas.get_attribute('width'));height=int(canvas.get_attribute('height'))
        click_points(bone,[(.25*width,.2*height),(.25*width,.35*height),(.25*width,.85*height),(.75*width,.2*height),(.75*width,.4*height),(.75*width,.85*height)])
        page.get_by_text('Draft CEJ/crest/apex ratio:',exact=False).wait_for(timeout=30000)
        text=page.get_by_text('Draft CEJ/crest/apex ratio:',exact=False).inner_text()
        values=list(map(float,re.findall(r'[0-9]+[.][0-9]+',text)));assert abs(values[0]-23.1)<1 and abs(values[1]-30.8)<1,text
        label=page.get_by_text('I reviewed these six anatomical landmarks and the measurement definition',exact=True)
        label.evaluate("el=>el.scrollIntoView({block:'center'})");label.click()
        page.get_by_role('button',name='Download reviewed bone measurements CSV',exact=True).wait_for(timeout=30000)
        assert not page.locator('[data-testid=stException]').count(),page.locator('[data-testid=stException]').all_inner_texts()
        assert 'Clinical AI Bone Level Assessment' not in page.locator('body').inner_text()
        print('PASS browser: drawn 37/38 boxes reach downstream tasks; X/Y/red=45deg; reviewed Winter angle=21.04deg; reviewed bone ratios=23.1/30.8%; no heuristic disease diagnosis')
        browser.close()


if __name__=='__main__':
    main()
