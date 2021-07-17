import os

import csv
import xml.etree.cElementTree as ET
from lxml import etree
label_dict = {
    'NAME': [
        'DOCTOR',
        'PATIENT',
        'USERNAME',
    ],
    'CONTACT': [
        'EMAIL',
        'FAX',
        'PHONE'
    ],
    'LOCATION': [
        'LOCATION',
        'HOSPITAL',
        'DEPARTMENT',
        'ORGANIZATION',
        'ROOM',
        'URL',
        'STREET',
        'STATE',
        'CITY',
        'COUNTRY',
        'ZIP',
        'LOCATION-OTHER',
    ],
    'DATE': [
        'DATE'
    ],
    'AGE': ['AGE'],
    'PROFESSION': ['PROFESSION'],
    'ID': [
        'BIOID',
        'DEVICE',
        'HEALTHPLAN',
        'IDNUM',
        'MEDICALRECORD',
    ],
}

reverse_label_dict = {
    
}
for parent, children in label_dict.items():
    for child in children:
        reverse_label_dict[child] = parent

DATA_PATH = 'data/output/bert-hsa'

pred_files = os.listdir(DATA_PATH + '/preds')
for filename in pred_files:
    pred_file = os.path.join(
        DATA_PATH, 'preds', filename
    )
    file_id = filename.split('.')[0]
    print(filename)
    txt_file = os.path.join(
        DATA_PATH, 'xml-test', file_id+'.xml', 
    )
    txt = etree.parse(txt_file).find('TEXT').text

    if not pred_file.endswith('.pred'):
        continue
    # root = ET.Element("deid")
    root = etree.Element("deid")
    doc = etree.Element("TAGS")
    root.append(doc)
    with open(pred_file) as pred, open(txt_file) as text_file:
        text_ele = etree.Element('TEXT')
        text_ele.text = etree.CDATA(txt)
        root.append(text_ele)
    
        preds = csv.reader(pred, delimiter=',')
        headers = next(preds, None)
        last_stop = 0
        last_label = None
        tags = []
        for row in preds:
            _, _, start_str, end_str, entity, label, _ = row
            start = int(start_str)
            stop = int(end_str)
            if start < last_stop:
                continue
            if (start == last_stop or (start == last_stop+1 and txt[start-1] == ' ')) and label == last_label:
                tags[-1]['end'] = end_str
            else:
                tags.append({
                    'label': label,
                    'start': start_str,
                    'end': end_str 
                })
            last_stop = stop
            last_label = label

        for tag in tags:
            start = int(tag['start'])
            end = int(tag['end'])
            child = etree.SubElement(doc, reverse_label_dict[tag['label']], text=txt[start:end], start=tag['start'], end=tag['end'], TYPE=tag['label'])
            # doc.append(child)

    tree = etree.ElementTree(root)
    tree.write(f"{DATA_PATH}/xml/{file_id}.xml", pretty_print=True)
