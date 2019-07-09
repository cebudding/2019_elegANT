from .dialog_box import DialogBox
from .button_build_ant import BuildAntButton
from .button_ants_dialog import AntsDialogButton


class DialogBoxAddAnts(DialogBox):
    def __init__(self, view, identifier, active=False, name="Dialog_Box"):
        super(DialogBoxAddAnts, self).__init__(view, identifier, x=0.03, y=0.87,
                                               width=0.02, height=0.10)
        self.active = active
        self.identifier = identifier
        self.name = name
        self.buttons = {}
        self.color = self.view.usercolor
        print(self.color)
        self.name = "_".join((str(c) for c in self.color))
        self.ant_types = [
            'scout',
            'worker',
            'soldier'
        ]
        self.set_buttons()

    def set_buttons(self):
        for idx, ant_type in enumerate(self.ant_types):
            build_ant_button = BuildAntButton(
                ant_type,
                self.view,
                f"build_{ant_type}", 0.10 + (idx * 0.10), 0.87, 0.05, 0.09, -1, self.color,
                self.color, 'square', True, f"src/view/images/{self.name}_build_{ant_type}.png"
            )

            if ant_type == 'scout':
                build_ant_button.on("click", lambda: self.view.event_dict.update({
                    'build_scout': ('build_scout',)
                }))
            if ant_type == 'worker':
                build_ant_button.on("click", lambda: self.view.event_dict.update({
                    'build_worker': ('build_worker',)
                }))
            if ant_type == 'soldier':
                build_ant_button.on("click", lambda: self.view.event_dict.update({
                    'build_soldier': ('build_soldier',)
                }))
            self.buttons[build_ant_button.identifier] = build_ant_button

    def toggle(self):
        self.active = not self.active

    def draw(self):
        super(DialogBoxAddAnts, self).draw()
        build_ants_button = AntsDialogButton(self.view, "show_build_ants", 0.02, 0.87, 0.05, 0.09, -1, self.color,
                                             self.color, 'square')

        build_ants_button.on("click", lambda: self.view.event_dict.update({
            "show_build_ants": (build_ants_button,)
        }))

        self.view.add_element(build_ants_button)

        if self.active:
            for identifier, button in self.buttons.items():
                self.view.add_element(button)
        else:
            for identifier in self.buttons.keys():
                self.view.remove_element(identifier)
