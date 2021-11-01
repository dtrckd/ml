from pymake import ExpeFormat, Spec, Script, get_pymake_settings
import subprocess
import os


class Cmd(ExpeFormat):

    def vi(self, *args):
        return self.vim(*args)

    def vim(self, *args):

        if len(args) > 0:
            # seek for script
            script = args[0]
            script = Script.find(script)
            module = script['module']
            script_folder = get_pymake_settings('default_script')
            script_folder ='/'.join(script_folder.split('.')[1:])
            fn = os.path.join(script_folder, module.split('.')[-1] + '.py')
        else:
            # seek for design
            spec = self.expe['_expe_name']
            spec = Spec.find(spec)
            module = spec['module_name']
            spec_folder = get_pymake_settings('default_spec')
            spec_folder ='/'.join(spec_folder.split('.')[1:])
            fn = os.path.join(spec_folder, module.split('.')[-2] + '.py')

        #subprocess.call(['vim', '-p']+list(args))
        subprocess.call(['vim', fn])

        #print(args, 'edited.')
        print(fn, 'edited.')
