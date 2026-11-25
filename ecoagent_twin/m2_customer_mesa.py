"""Activate simulated customer responses at a declared offer-response time."""

from collections.abc import Iterable

from .m2_offers import OfferLedger, OfferOption, UtilityProfile, paired_draw


def respond(
    ledger: OfferLedger, offers: Iterable[tuple[OfferOption, int]], profile: UtilityProfile,
    *, scenario: str, day: int, seed: int, response_min: int,
) -> None:
    import mesa

    class Recipient(mesa.Agent):
        def __init__(self, model, option: OfferOption, credit_cents: int):
            super().__init__(model)
            self.option = option
            self.credit_cents = credit_cents

        def respond_to_offer(self) -> None:
            draw = paired_draw(day, seed, self.option.parcel_id, scenario=scenario)
            accepted = draw < profile.accept_probability(self.option, self.credit_cents)
            ledger.resolve(self.option.parcel_id, now_min=response_min, accepted=accepted)

    model = mesa.Model(seed=seed)
    for option, credit_cents in offers:
        Recipient(model, option, credit_cents)
    model.agents.do("respond_to_offer")
