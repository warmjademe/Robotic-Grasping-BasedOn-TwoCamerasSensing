"""Expand archived FANUC macro and mount the same Panda tool used by old plans."""
from pathlib import Path
import copy, hashlib, json, math, re, xml.etree.ElementTree as ET
RUN=Path(__file__).resolve().parents[1]
STUDY=RUN.parent
PROJECT=STUDY.parent
BASE=Path('/home/qyb/RESEARCH/baseline_graspgen')
X='{http://wiki.ros.org/xacro}'
def main():
    source=STUDY/'experiments/robot_assets/fanuc_description/fanuc_m20_description/urdf/m20_25_18d_urdf_macro.xacro'
    root=ET.parse(source).getroot(); macro=root.find(X+'macro')
    props={x.attrib['name']:x.attrib['value'] for x in macro.findall(X+'property')}
    robot=ET.Element('robot',name='fanuc_m20id25_panda_evaluation')
    def subst(s):
        s=s.replace('${prefix}','')
        for k,v in props.items():s=s.replace('${'+k+'}',v)
        s=re.sub(r'\$\{radians\(\s*([\d.\-]+)\)\}',lambda m:str(math.radians(float(m[1]))),s)
        return s
    for element in macro:
        if element.tag not in ['link','joint']:continue
        e=copy.deepcopy(element)
        if e.attrib.get('name') in ['${prefix}wbase','${prefix}base_link-wbase','${prefix}base_joint','${prefix}child_joint']:continue
        for node in e.iter():
            for k,v in node.attrib.items():node.set(k,subst(v))
        for mesh in e.findall('.//mesh'):
            name=Path(mesh.attrib['filename']).stem
            f=source.parent.parent/'meshes/m20_25_18d/collision'/(name+'.stl')
            assert f.exists();mesh.set('filename',str(f))
        robot.append(e)
    panda_source=BASE/'upstream/GraspGen/assets/franka/franka_panda.urdf'
    panda=ET.parse(panda_source).getroot()
    for name in ['panda_hand','panda_leftfinger','panda_rightfinger']:
        e=copy.deepcopy(panda.find("link[@name='"+name+"']"))
        for visual in e.findall('visual'):e.remove(visual)
        collision=e.find('collision');visual=copy.deepcopy(collision);visual.tag='visual';e.append(visual)
        for mesh in e.findall('.//mesh'):mesh.set('filename',str(panda_source.parent/mesh.attrib['filename']))
        robot.append(e)
    joint=ET.SubElement(robot,'joint',name='tool_mount',type='fixed')
    ET.SubElement(joint,'parent',link='fanuc_flange');ET.SubElement(joint,'child',link='panda_hand')
    ET.SubElement(joint,'origin',xyz='0 0 .04',rpy='0 0 0')
    for name in ['panda_finger_joint1','panda_finger_joint2']:
        e=copy.deepcopy(panda.find("joint[@name='"+name+"']"))
        mimic=e.find('mimic')
        if mimic is not None:e.remove(mimic)
        robot.append(e)
    out=RUN/'assets';out.mkdir(exist_ok=True)
    ET.indent(robot)
    ET.ElementTree(robot).write(out/'fanuc_panda.urdf',encoding='unicode',xml_declaration=True)
    provenance=dict(fanuc_macro=str(source.relative_to(STUDY)),fanuc_macro_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        panda_urdf_sha256=hashlib.sha256(panda_source.read_bytes()).hexdigest(),
        generated_sha256=hashlib.sha256((out/'fanuc_panda.urdf').read_bytes()).hexdigest(),
        joint_conversion='serial J3 = controller J2 + controller J3; all others unchanged',
        tool_mount_z_m=.04,visual_meshes='archived collision meshes',
        controls='PhysX joint drives, not a FANUC virtual controller')
    (out/'robot_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance))
if __name__=='__main__':main()
