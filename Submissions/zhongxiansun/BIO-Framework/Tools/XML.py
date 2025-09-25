import os
import xmltodict
import copy
import numpy as np
import datetime
class XMLfile():
    def __init__(self, id=0, round_idx=0, stop_volt1=4.5, stop_volt2=3.6):
           
        policy_file = 'policies_all.csv'
        xml_file_path = 'Others/Reference/pulse_sample.xml'
        file_dir = 'Tools/File4test/'

        self.id = id
        self.policy_file = os.path.join(policy_file)
        self.xml_file_path = os.path.join(xml_file_path)
        self.xml_folder = os.path.join(file_dir, f'budget@runs_{round_idx}')
        self.stop_volt1 = stop_volt1
        self.stop_volt2 = stop_volt2
        os.makedirs(self.xml_folder, exist_ok=True)
        
        self.valid_policies = self.get_valid_policies()
        self.xml_temp = self.get_xml_temp()

        self.xml_case = {'root':{'config':{}}}

        self.def_materials()
        self.initial()

    def def_materials(self):
        specific_capacity = int(165)  # Specific_capacity of LiNi0.5Co0.2Mn0.3O2
        Area = 3.1416 * 0.6 ** 2  # Diameter 12 mm
        load = 10.6  # Loading of the electrode
        self.Q = round(specific_capacity * Area * load * 0.95 * 0.001, 2)  # Capacity
        self.I0 = self.Q / 1  # mA

        self.Rate0 = 0.5


    def initial(self):
        for non_Step, details in self.xml_temp['root']['config'].items():
            if non_Step != 'Step_Info' :
                self.xml_case['root']['config'][non_Step] = details
            else:
                self.xml_case['root']['config'][non_Step] = {}
        stepinfo = self.xml_temp['root']['config']['Step_Info']
        self.head = stepinfo
        self.rest = stepinfo['Step1']
        self.charging = stepinfo['Step2']
        self.cv = stepinfo['Step3']
        self.discharging = stepinfo['Step4']
        self.loop = stepinfo['Step5']
        self.pulse = stepinfo['Step6']
        self.jump = stepinfo['Step8']

    def get_step_no(self):
        return len(self.xml_case['root']['config']['Step_Info'].keys()) + 1
    
    def append_step(self, step_template, step_name, **kwargs):
        step_no = self.get_step_no()
        Step_ = f'Step{step_no}'
        step_data = copy.deepcopy(step_template)
        step_data['@Step_ID'] = step_no
        for key, value in kwargs.items():
            keys = key.split('/')
            current_level = step_data
            for k in keys[:-1]:
                current_level = current_level.get(k, {})
            current_level[keys[-1]] = value 
        self.xml_case['root']['config']['Step_Info'][Step_] = step_data

    def append_pulse(self, policy, baseline,LOOP):

        Rate1, Rate2, Ton1, Ton2, Toff1, Toff2 = policy

        Stop_Volt1 = 3.78
        Stop_Volt2 = 3.95

        def Active():
            self.append_step(self.rest, "initial_rest",
                **{
                    'Record/Main/Time/@Value' : float(30000) ,# ms
                    'Limit/Main/Time/@Value' : float(36000*1000) 
                })
            loop = self.get_step_no()
            self.append_step(self.charging, "active_charge",
                **{
                    'Record/Main/Time/@Value' : float(30000) ,
                    'Limit/Main/Rate/@Value' : float(0.2) ,
                    'Limit/Main/Stop_Volt/@Value' : float(self.stop_volt1*10000) 
                })
            self.append_step(self.discharging, "active_discharge",
                **{
                    'Record/Main/Time/@Value' : float(30000) ,
                    'Limit/Main/Rate/@Value' : float(0.2) ,
                    'Limit/Main/Stop_Volt/@Value' : float(self.stop_volt2*10000) 
                })
            self.append_step(self.loop, "Loop",
                **{
                    'Limit/Other/Start_Step/@Value': int(loop),
                    'Limit/Other/Cycle_Count/@Value': int(3)
                })
        
        def PC(Cur, Ton, Toff, Stop_Volt):
            loopi = self.get_step_no()
            chock = loopi + 4
            self.append_step(self.pulse, "PC-on", 
                **{
                    'Record/Main/Time/@Value' : float(100) ,
                    'Limit/Main/Rate/@Value' : float(Cur) ,
                    'Limit/Main/Time/@Value' : float(Ton * 1000) ,
                    'Limit/Main/Stop_Volt/@Value' : float(self.stop_volt1*10000)       
                })

            self.append_step(self.rest, "PC-off1",
                **{
                    'Record/Main/Time/@Value' : float(100) ,# ms
                    'Limit/Main/Time/@Value' : float(Toff * 1000 - 100) 
                })
            self.append_step(self.jump, "Judge",
                **{
                    'Record/Main/Time/@Value' : float(100) ,# ms
                    'Limit/Main/Time/@Value' : float(100) ,# ms, 
                    'Limit/Other/Cnd1/@Jump_Line' : int(chock) ,
                    'Limit/Other/Cnd1/@Value' : float(Stop_Volt * 10000) 
                })
            self.append_step(self.loop, "Loop",
                **{
                    'Limit/Other/Start_Step/@Value': int(loopi),
                    'Limit/Other/Cycle_Count/@Value': int(65535)
                })
        
        def Addition(loopk):
            
            self.append_step(self.charging, "charge to the end",
                        **{
                            'Record/Main/Time/@Value' : float(100) ,
                            'Limit/Main/Rate/@Value' : float(baseline) ,
                            'Limit/Main/Stop_Volt/@Value' : float(self.stop_volt1*10000)
                        })
            
            self.append_step(self.discharging, "discharge to the end",
                        **{
                            'Record/Main/Time/@Value' : float(30000) ,
                            'Limit/Main/Rate/@Value' : float(baseline) ,
                            'Limit/Main/Stop_Volt/@Value' : float(self.stop_volt2*10000)
                        })
            self.append_step(self.loop, "Loop",
                        **{
                            'Limit/Other/Start_Step/@Value': int(loopk),
                            'Limit/Other/Cycle_Count/@Value': int(LOOP)
                        })

        Active()
        loopk = self.get_step_no()
        PC(Rate1 * 0.5, Ton1, Toff1, Stop_Volt1)
        PC(Rate2 * 0.5, Ton2, Toff2, Stop_Volt2)
        Addition(loopk)

        return self.xml_case
    
    def run(self, LOOP):
        date = datetime.datetime.now()
        datelis = date.strftime("%Y%m%d%H%M%S")
        self.xml_case['root']['config']['@date'] = f'{datelis}'
        self.xml_case['root']['config']['Head_Info']['Creator']['@Value'] = "SZX"
        policy = self.valid_policies[self.id]
        case = self.append_pulse(policy, self.Rate0, LOOP)
        case['root']['config']['Step_Info']['@Num'] = len(case['root']['config']['Step_Info'])

        return(policy, case)
    
    def to_xml(self, index, LOOP=50):
        _, policy_file = self.run(LOOP)
        policy_file['root']['config']['Head_Info']['Remark']['@Value'] = f"PULSE{index}"
        
        xml_file_path = os.path.join(self.xml_folder, f'policy_{index}.xml')
        with open(xml_file_path, 'w') as fr: 
            fr.write(xmltodict.unparse(policy_file, pretty=True))
    
    def get_valid_policies(self):
        policies = np.genfromtxt(self.policy_file, delimiter=',', skip_header=1)
        # np.random.shuffle(policies)
        return policies[:, :6]
    def get_xml_temp(self):
        with open(self.xml_file_path, 'r', encoding='utf-8') as xf:
            xml_temp = xmltodict.parse(xf.read())
        return xml_temp

